import os
import base64
import requests
from pathlib import Path
from typing import Optional, Union
from PIL import Image
import io
import boto3
from botocore.config import Config

class Step1X3DClient:
    def __init__(
        self, 
        base_url: str = "http://localhost:8001",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: str = "us-west-2"
    ):
        """Initialize the client with the API base URL and AWS credentials"""
        self.base_url = base_url.rstrip('/')
        
        # Set up AWS credentials
        if aws_access_key_id and aws_secret_access_key:
            os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key_id
            os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_access_key
            os.environ['AWS_DEFAULT_REGION'] = aws_region
        
        # Initialize S3 client with retry configuration
        self.s3_client = boto3.client(
            's3',
            config=Config(
                retries=dict(
                    max_attempts=3
                )
            )
        )
        
    def _encode_image_to_base64(self, image_path: str) -> str:
        """Convert an image file to base64 string"""
        with open(image_path, 'rb') as image_file:
            return f"data:image/png;base64,{base64.b64encode(image_file.read()).decode()}"
    
    def _prepare_input(self, input_data: Union[str, Path]) -> str:
        """Prepare input data for API request"""
        if isinstance(input_data, (str, Path)):
            input_str = str(input_data)
            if input_str.startswith('s3://'):
                # Verify S3 access before sending to API
                try:
                    bucket, key = self._parse_s3_url(input_str)
                    self.s3_client.head_object(Bucket=bucket, Key=key)
                    return input_str
                except Exception as e:
                    raise ValueError(f"Error accessing S3 object: {str(e)}")
            elif os.path.isfile(input_str):
                return self._encode_image_to_base64(input_str)
            else:
                raise ValueError(f"Invalid input path: {input_str}")
        else:
            raise ValueError("Input must be a file path or S3 URL")
    
    def _parse_s3_url(self, url: str) -> tuple[str, str]:
        """Parse S3 URL into bucket and key"""
        if not url.startswith('s3://'):
            raise ValueError("Invalid S3 URL format")
        path = url[5:]  # Remove 's3://'
        bucket = path.split('/')[0]
        key = '/'.join(path.split('/')[1:])
        return bucket, key

    def generate_geometry(
        self,
        input_image: Union[str, Path],
        output_uri: str,
    ) -> dict:
        """
        Generate geometry from an input image
        
        Args:
            input_image: Path to input image file or S3 URL
            output_uri: S3 URI where the output should be saved (e.g., s3://bucket/prefix/filename.glb)
            
        Returns:
            dict: Response containing the S3 URL of the generated geometry
        """
        input_data = self._prepare_input(input_image)
        
        response = requests.post(
            f"{self.base_url}/generate/geometry",
            json={
                "input_image": input_data,
                "output_uri": output_uri
            }
        )
        response.raise_for_status()
        return response.json()

    def generate_geometry_with_label(
        self,
        input_image: Union[str, Path],
        output_uri: str,
        symmetry: str = "x",
        edge_type: str = "sharp"
    ) -> dict:
        """
        Generate geometry with label control from an input image
        
        Args:
            input_image: Path to input image file or S3 URL
            output_uri: S3 URI where the output should be saved
            symmetry: Symmetry type (x, y, z)
            edge_type: Edge type (sharp, smooth)
            
        Returns:
            dict: Response containing the S3 URL of the generated geometry
        """
        input_data = self._prepare_input(input_image)
        
        response = requests.post(
            f"{self.base_url}/generate/geometry-with-label",
            json={
                "input_image": input_data,
                "output_uri": output_uri,
                "symmetry": symmetry,
                "edge_type": edge_type
            }
        )
        response.raise_for_status()
        return response.json()

    def generate_texture(
        self,
        input_image: Union[str, Path],
        input_geometry: Union[str, Path],
        output_uri: str
    ) -> dict:
        """
        Generate textured model from an input image and geometry
        
        Args:
            input_image: Path to input image file or S3 URL
            input_geometry: Path to input geometry file or S3 URL
            output_uri: S3 URI where the output should be saved
            
        Returns:
            dict: Response containing the S3 URL of the generated textured model
        """
        input_image_data = self._prepare_input(input_image)
        input_geometry_data = self._prepare_input(input_geometry)
        
        response = requests.post(
            f"{self.base_url}/generate/texture",
            json={
                "input_image": input_image_data,
                "input_geometry": input_geometry_data,
                "output_uri": output_uri
            }
        )
        response.raise_for_status()
        return response.json()

def main():
    # Example usage
    client = Step1X3DClient()
    
    # Example 1: Generate geometry from a local image file
    try:
        result = client.generate_geometry(
            input_image="examples/images/001.png",
            output_uri="s3://genies-ml-rnd/step1x3d/geometry.glb"
        )
        print("Generated geometry:", result["s3_url"])
    except Exception as e:
        print(f"Error generating geometry: {e}")
    
    # # Example 2: Generate geometry with label from an S3 image
    # try:
    #     result = client.generate_geometry_with_label(
    #         input_image="s3://my-bucket/input/image.png",
    #         output_uri="s3://my-bucket/output/geometry_label.glb",
    #         symmetry="y",
    #         edge_type="smooth"
    #     )
    #     print("Generated geometry with label:", result["s3_url"])
    # except Exception as e:
    #     print(f"Error generating geometry with label: {e}")
    
    # Example 3: Generate textured model
    try:
        result = client.generate_texture(
            input_image="examples/images/001.png",
            input_geometry="s3://genies-ml-rnd/step1x3d/geometry.glb",
            output_uri="s3://genies-ml-rnd/step1x3d/textured.glb"
        )
        print("Generated textured model:", result["s3_url"])
    except Exception as e:
        print(f"Error generating textured model: {e}")

if __name__ == "__main__":
    main()
