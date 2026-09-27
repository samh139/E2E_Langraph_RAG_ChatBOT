import os
from ollama import Client
from pathlib import Path 
from dotenv import load_dotenv
import base64

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)

class OllamaClient:

    def __init__(self):
        self.client = Client(host="http://localhost:11434")

    def analyze_vision(self, image_bytes: bytes, user_prompt: str, model_name: str = "minicpm-v:latest") -> str:
        """
        Executes a cloud-accelerated vision transcription using Ollama Cloud.
        Passes raw images seamlessly along with text instructions.
        """
        if not image_bytes:
            return ""

        try:
            # Convert binary data to string format for transportation over the API network
            base64_image = base64.b64encode(image_bytes).decode('utf-8')

            # Dynamic config targeting your chosen Qwen cloud deployment
            kwargs = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": user_prompt,
                        "images": [base64_image] # Ollama SDK handles lists of base64 images
                    }
                ]
            }

            # Send requests directly to https://ollama.com hosting pools
            response = self.client.chat(**kwargs)
            return response["message"]["content"]

        except Exception as err:
            print(f"[Ollama Cloud Vision Error] Request failed: {err}")
            return ""
