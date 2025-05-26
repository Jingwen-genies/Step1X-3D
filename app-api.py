import os
import torch
import trimesh
import shutil
import boto3
import backoff
import base64
import io
from pathlib import Path
from typing import Optional, Union
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from step1x3d_texture.pipelines.step1x_3d_texture_synthesis_pipeline import Step1X3DTexturePipeline
from step1x3d_geometry.models.pipelines.pipeline_utils import reduce_face, remove_degenerate_face
from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline
import uvicorn
from huggingface_hub import snapshot_download
from botocore.exceptions import ClientError, ConnectionError, ConnectTimeoutError
from botocore.config import Config

# Constants
MODEL_CACHE_DIR = Path("model_cache")
TEMP_DIR = Path("temp")
OUTPUT_DIR = Path("output")

# Create necessary directories
MODEL_CACHE_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize S3 client with retry configuration
s3_client = boto3.client(
    's3',
    config=Config(
        retries=dict(
            max_attempts=3
        )
    )
)

def parse_s3_url(url: str) -> tuple[str, str]:
    """Parse S3 URL into bucket and key"""
    if not url.startswith('s3://'):
        raise ValueError("Invalid S3 URL format")
    path = url[5:]  # Remove 's3://'
    bucket = path.split('/')[0]
    key = '/'.join(path.split('/')[1:])
    return bucket, key

@backoff.on_exception(
    backoff.expo,
    (ClientError, ConnectionError, ConnectTimeoutError),
    max_tries=5,
    max_time=30
)
def upload_to_s3(file_path: str, bucket: str, prefix: str) -> str:
    """Upload a file to S3 with exponential backoff retry"""
    try:
        # Verify bucket exists and is accessible
        s3_client.head_bucket(Bucket=bucket)
        
        # Upload file
        s3_client.upload_file(file_path, bucket, prefix)
        return f"s3://{bucket}/{prefix}"
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            raise HTTPException(status_code=404, detail=f"S3 bucket not found: {bucket}")
        elif error_code == '403':
            raise HTTPException(status_code=403, detail=f"Access denied to S3 bucket: {bucket}")
        else:
            raise HTTPException(status_code=500, detail=f"S3 error: {str(e)}")
    except Exception as e:
        print(f"Error uploading to S3 (attempt will be retried): {str(e)}")
        raise

@backoff.on_exception(
    backoff.expo,
    (ClientError, ConnectionError, ConnectTimeoutError),
    max_tries=5,
    max_time=30
)
def download_from_s3(url: str, local_path: str) -> str:
    """Download a file from S3 with exponential backoff retry"""
    try:
        bucket, key = parse_s3_url(url)
        
        # Verify object exists and is accessible
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                raise HTTPException(status_code=404, detail=f"S3 object not found: {url}")
            elif error_code == '403':
                raise HTTPException(status_code=403, detail=f"Access denied to S3 object: {url}")
            else:
                raise HTTPException(status_code=500, detail=f"S3 error: {str(e)}")
        
        # Download file
        s3_client.download_file(bucket, key, local_path)
        return local_path
    except Exception as e:
        print(f"Error downloading from S3 (attempt will be retried): {str(e)}")
        raise

def save_base64_image(base64_str: str, output_path: str) -> str:
    """Save base64 encoded image to file"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        # Decode base64 string
        image_data = base64.b64decode(base64_str)
        
        # Save to file
        with open(output_path, 'wb') as f:
            f.write(image_data)
        
        return output_path
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

def get_image_from_input(
    input_data: Union[UploadFile, str],
    temp_dir: Path,
    file_name: str
) -> str:
    """Handle different types of image input and return local file path"""
    if isinstance(input_data, UploadFile):
        # Handle file upload
        input_path = temp_dir / file_name
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(input_data.file, buffer)
        return str(input_path)
    elif input_data.startswith('s3://'):
        # Handle S3 URL
        input_path = temp_dir / file_name
        return download_from_s3(input_data, str(input_path))
    elif input_data.startswith('data:image') or input_data.startswith('data:application'):
        # Handle base64
        input_path = temp_dir / file_name
        return save_base64_image(input_data, str(input_path))
    else:
        raise HTTPException(status_code=400, detail="Invalid input format. Must be file upload, S3 URL, or base64 string")

class Step1X3DApp:
    def __init__(self):
        # Download and cache models if not already present
        self._download_models()
        
        # Initialize models
        print("Loading geometry model...")
        self.geometry_pipeline = Step1X3DGeometryPipeline.from_pretrained(
            str(MODEL_CACHE_DIR / "Step1X-3D-Geometry-1300m"),
            local_files_only=True
        ).to("cuda")
        
        print("Loading geometry label model...")
        self.geometry_label_pipeline = Step1X3DGeometryPipeline.from_pretrained(
            str(MODEL_CACHE_DIR / "Step1X-3D-Geometry-Label-1300m"),
            local_files_only=True
        ).to("cuda")
        
        print("Loading texture model...")
        self.texture_pipeline = Step1X3DTexturePipeline.from_pretrained(
            "stepfun-ai/Step1X-3D",
            subfolder="Step1X-3D-Texture"
        )
        
        # Initialize generator
        self.generator = torch.Generator(device="cuda")
        self.generator.manual_seed(2025)
        
        print("All models loaded successfully!")

    def _download_models(self):
        """Download and cache models if they don't exist locally"""
        if not (MODEL_CACHE_DIR / "Step1X-3D-Geometry-1300m").exists():
            print("Downloading geometry model...")
            snapshot_download(
                "stepfun-ai/Step1X-3D",
                local_dir=MODEL_CACHE_DIR,
                local_dir_use_symlinks=False,
                repo_type="model",
                allow_patterns=["Step1X-3D-Geometry-1300m/*"]
            )
        
        if not (MODEL_CACHE_DIR / "Step1X-3D-Geometry-Label-1300m").exists():
            print("Downloading geometry label model...")
            snapshot_download(
                "stepfun-ai/Step1X-3D",
                local_dir=MODEL_CACHE_DIR,
                local_dir_use_symlinks=False,
                repo_type="model",
                allow_patterns=["Step1X-3D-Geometry-Label-1300m/*"]
            )
        
        if not (MODEL_CACHE_DIR / "Step1X-3D-Texture").exists():
            print("Downloading texture model...")
            snapshot_download(
                "stepfun-ai/Step1X-3D",
                local_dir=MODEL_CACHE_DIR,
                local_dir_use_symlinks=False,
                repo_type="model",
                allow_patterns=["Step1X-3D-Texture/*"]
            )

    def process_geometry(self, input_image_path, save_glb_path, bucket: str, prefix: str):
        """Process input image to generate geometry"""
        try:
            out = self.geometry_pipeline(
                input_image_path, 
                guidance_scale=7.5, 
                num_inference_steps=50, 
                generator=self.generator
            )
            
            os.makedirs(os.path.dirname(save_glb_path), exist_ok=True)
            out.mesh[0].export(save_glb_path)
            
            # Upload to S3 - use the prefix directly without adding temp.glb
            s3_url = upload_to_s3(save_glb_path, bucket, prefix)
            
            return s3_url
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing geometry: {str(e)}")

    def process_geometry_with_label(self, input_image_path, save_glb_path, bucket: str, prefix: str, symmetry="x", edge_type="sharp"):
        """Process input image to generate geometry with label control"""
        try:
            out = self.geometry_label_pipeline(
                input_image_path,
                label={"symmetry": symmetry, "edge_type": edge_type},
                guidance_scale=7.5,
                octree_resolution=384,
                max_facenum=400000,
                num_inference_steps=50,
                generator=self.generator
            )
            
            os.makedirs(os.path.dirname(save_glb_path), exist_ok=True)
            out.mesh[0].export(save_glb_path)
            
            # Upload to S3 - use the prefix directly without adding _label.glb
            s3_url = upload_to_s3(save_glb_path, bucket, prefix)
            
            return s3_url
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing geometry with label: {str(e)}")

    def process_texture(self, input_image_path, input_glb_path, save_glb_path, bucket: str, prefix: str):
        """Process input image and geometry to generate textured model"""
        try:
            mesh = trimesh.load(input_glb_path)
            mesh = remove_degenerate_face(mesh)
            mesh = reduce_face(mesh)
            textured_mesh = self.texture_pipeline(input_image_path, mesh, seed=2025)
            
            os.makedirs(os.path.dirname(save_glb_path), exist_ok=True)
            textured_mesh.export(save_glb_path)
            
            # Upload to S3 - use the prefix directly without adding _texture.glb
            s3_url = upload_to_s3(save_glb_path, bucket, prefix)
            
            return s3_url
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing texture: {str(e)}")

# Initialize FastAPI app
app = FastAPI(
    title="Step1X3D API",
    description="API for generating 3D models from images using Step1X3D",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the application
step1x_app = Step1X3DApp()

@app.post("/generate/geometry")
async def generate_geometry(
    input_image: Union[UploadFile, str] = Body(..., description="Input image as file upload, S3 URL, or base64 string"),
    output_uri: str = Body(..., description="S3 URI where the output should be saved (e.g., s3://bucket/prefix/filename.glb)")
):
    """Generate geometry from an input image"""
    try:
        # Parse output URI
        output_bucket, output_prefix = parse_s3_url(output_uri)
        
        # Handle input image
        input_path = get_image_from_input(input_image, TEMP_DIR, "input_image.png")
        
        # Generate output path
        output_path = OUTPUT_DIR / "temp.glb"
        
        # Process the image and get S3 URL
        s3_url = step1x_app.process_geometry(
            input_path, 
            str(output_path),
            bucket=output_bucket,
            prefix=output_prefix
        )
        
        # Clean up
        os.remove(input_path)
        os.remove(output_path)
        
        return {"s3_url": s3_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/geometry-with-label")
async def generate_geometry_with_label(
    input_image: Union[UploadFile, str] = Body(..., description="Input image as file upload, S3 URL, or base64 string"),
    output_uri: str = Body(..., description="S3 URI where the output should be saved (e.g., s3://bucket/prefix/filename.glb)"),
    symmetry: str = Body("x", description="Symmetry type (x, y, z)"),
    edge_type: str = Body("sharp", description="Edge type (sharp, smooth)")
):
    """Generate geometry with label control from an input image"""
    try:
        # Parse output URI
        output_bucket, output_prefix = parse_s3_url(output_uri)
        
        # Handle input image
        input_path = get_image_from_input(input_image, TEMP_DIR, "input_image.png")
        
        # Generate output path
        output_path = OUTPUT_DIR / "temp_geometry_label.glb"
        
        # Process the image and get S3 URL
        s3_url = step1x_app.process_geometry_with_label(
            input_path, 
            str(output_path),
            bucket=output_bucket,
            prefix=output_prefix,
            symmetry=symmetry,
            edge_type=edge_type
        )
        
        # Clean up
        os.remove(input_path)
        os.remove(output_path)
        
        return {"s3_url": s3_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/texture")
async def generate_texture(
    input_image: Union[UploadFile, str] = Body(..., description="Input image as file upload, S3 URL, or base64 string"),
    input_geometry: Union[UploadFile, str] = Body(..., description="Input geometry as file upload, S3 URL, or base64 string"),
    output_uri: str = Body(..., description="S3 URI where the output should be saved (e.g., s3://bucket/prefix/filename.glb)")
):
    """Generate textured model from an input image and geometry"""
    try:
        # Parse output URI
        output_bucket, output_prefix = parse_s3_url(output_uri)
        
        # Handle input files
        image_path = get_image_from_input(input_image, TEMP_DIR, "input_image.png")
        geometry_path = get_image_from_input(input_geometry, TEMP_DIR, "input_geometry.glb")
        
        # Generate output path
        output_path = OUTPUT_DIR / "temp_textured.glb"
        
        # Process the files and get S3 URL
        s3_url = step1x_app.process_texture(
            image_path, 
            geometry_path, 
            str(output_path),
            bucket=output_bucket,
            prefix=output_prefix
        )
        
        # Clean up
        os.remove(image_path)
        os.remove(geometry_path)
        os.remove(output_path)
        
        return {"s3_url": s3_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
