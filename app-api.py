import os
import torch
import trimesh
import shutil
import boto3
import backoff
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from step1x3d_texture.pipelines.step1x_3d_texture_synthesis_pipeline import Step1X3DTexturePipeline
from step1x3d_geometry.models.pipelines.pipeline_utils import reduce_face, remove_degenerate_face
from step1x3d_geometry.models.pipelines.pipeline import Step1X3DGeometryPipeline
import uvicorn
from huggingface_hub import snapshot_download
from botocore.exceptions import ClientError, ConnectionError, ConnectTimeoutError

# Constants
MODEL_CACHE_DIR = Path("model_cache")
TEMP_DIR = Path("temp")
OUTPUT_DIR = Path("output")

# Create necessary directories
MODEL_CACHE_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize S3 client
s3_client = boto3.client('s3')

@backoff.on_exception(
    backoff.expo,
    (ClientError, ConnectionError, ConnectTimeoutError),
    max_tries=5,
    max_time=30
)
def upload_to_s3(file_path: str, bucket: str, prefix: str):
    """Upload a file to S3 with exponential backoff retry"""
    try:
        s3_client.upload_file(file_path, bucket, prefix)
        return f"s3://{bucket}/{prefix}"
    except Exception as e:
        print(f"Error uploading to S3 (attempt will be retried): {str(e)}")
        raise

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
            
            # Upload to S3
            file_name = Path(save_glb_path).stem
            s3_file = f"{file_name}.glb"
            s3_url = upload_to_s3(save_glb_path, bucket, f"{prefix}/{s3_file}")
            
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
            
            # Upload to S3
            file_name = Path(save_glb_path).stem
            s3_file = f"{file_name}_label.glb"
            s3_url = upload_to_s3(save_glb_path, bucket, f"{prefix}/{s3_file}")
            
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
            
            # Upload to S3
            file_name = Path(save_glb_path).stem
            s3_file = f"{file_name}_texture.glb"
            s3_url = upload_to_s3(save_glb_path, bucket, f"{prefix/{s3_file}}")
            
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
    file: UploadFile = File(...),
    bucket: str = Form(...),
    prefix: str = Form(...)
):
    """Generate geometry from an input image"""
    try:
        # Save uploaded file temporarily
        input_path = TEMP_DIR / file.filename
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Use the class method directly - it handles the pipeline and saving
        s3_url = step1x_app.process_geometry(str(input_path), str(OUTPUT_DIR / "temp.glb"))
        
        # Upload to S3
        # s3_url = upload_to_s3(result_path, bucket, prefix)
        
        # Clean up
        os.remove(input_path)
        os.remove(s3_url)
        
        return {"s3_url": s3_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/geometry-with-label")
async def generate_geometry_with_label(
    file: UploadFile = File(...),
    bucket: str = Form(...),
    prefix: str = Form(...),
    symmetry: str = Form("x"),
    edge_type: str = Form("sharp")
):
    """Generate geometry with label control from an input image"""
    try:
        # Save uploaded file
        input_path = TEMP_DIR / file.filename
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Generate output path
        output_path = OUTPUT_DIR / f"{Path(file.filename).stem}_geometry_label.glb"
        
        # Process the image and get S3 URL
        s3_url = step1x_app.process_geometry_with_label(
            str(input_path), 
            str(output_path),
            bucket=bucket,
            prefix=prefix,
            symmetry=symmetry,
            edge_type=edge_type
        )
        
        # Clean up input file
        os.remove(input_path)
        
        return {"s3_url": s3_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/texture")
async def generate_texture(
    image_file: UploadFile = File(...),
    geometry_file: UploadFile = File(...),
    bucket: str = Form(...),
    prefix: str = Form(...)
):
    """Generate textured model from an input image and geometry"""
    try:
        # Save uploaded files
        image_path = TEMP_DIR / image_file.filename
        geometry_path = TEMP_DIR / geometry_file.filename
        
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image_file.file, buffer)
        with open(geometry_path, "wb") as buffer:
            shutil.copyfileobj(geometry_file.file, buffer)
        
        # Generate output path
        output_path = OUTPUT_DIR / f"{Path(image_file.filename).stem}_textured.glb"
        
        # Process the files and get S3 URL
        s3_url = step1x_app.process_texture(
            str(image_path), 
            str(geometry_path), 
            str(output_path),
            bucket=bucket,
            prefix=prefix
        )
        
        # Clean up input files
        os.remove(image_path)
        os.remove(geometry_path)
        
        return {"s3_url": s3_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
