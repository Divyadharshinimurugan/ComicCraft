from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()

prompt = """
Create a colorful cartoon comic panel.

A young college girl named Maya is standing at the entrance
of a magical forest. She has a backpack and looks curious.

The forest has glowing blue and purple plants.

Bright, friendly, clean comic-book illustration style.
No scary or violent elements.
"""

response = client.models.generate_content(
    model="gemini-3.1-flash-image",
    contents=prompt
)

for part in response.parts:
    if part.inline_data is not None:
        image = part.as_image()
        image.save("test_comic_panel.png")
        print("SUCCESS! Comic panel image generated.")
        print("Saved as test_comic_panel.png")