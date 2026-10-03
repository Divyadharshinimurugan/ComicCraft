import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

token = os.getenv("HF_TOKEN")

if not token:
    print("ERROR: HF_TOKEN missing from .env")
    raise SystemExit(1)

output_folder = Path("static") / "images"
output_folder.mkdir(parents=True, exist_ok=True)

try:
    print("Connecting to Hugging Face...")

    client = InferenceClient(
        api_key=token,
        provider="auto",
        timeout=120,
    )

    print("Generating test comic image...")

    image = client.text_to_image(
        prompt=(
            "A cheerful little robot helping a student in a library, "
            "colorful 2D comic book illustration, clean outlines, "
            "bright colors, single panel, no text"
        ),
        model="black-forest-labs/FLUX.1-schnell",
    )

    image.save(output_folder / "test_comic.png")

    print("SUCCESS: Image generated!")
    print("Saved to: static/images/test_comic.png")

except Exception as error:
    print("Image generation failed.")
    print("Error type:", type(error).__name__)
    print("Error:", str(error)[:700])