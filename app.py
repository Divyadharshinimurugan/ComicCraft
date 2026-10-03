from flask import Flask, render_template, request
from flask import send_file
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from google import genai
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

import time
import os
import re

# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

app = Flask(__name__)

# =========================================================
# GEMINI CLIENT
# =========================================================

gemini_client = genai.Client()

# =========================================================
# HUGGING FACE CLIENT
# =========================================================

hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise RuntimeError(
        "HF_TOKEN is missing from .env file."
    )

hf_client = InferenceClient(
    api_key=hf_token,
    provider="auto",
    timeout=300,
)

# =========================================================
# IMAGE FOLDER
# =========================================================

IMAGE_FOLDER = os.path.join(
    "static",
    "generated"
)

os.makedirs(
    IMAGE_FOLDER,
    exist_ok=True
)


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# GENERATE COMIC STORY
# =========================================================

@app.route(
    "/generate",
    methods=["POST"]
)
def generate():

    try:

        story = request.json.get(
            "story",
            ""
        ).strip()

        if not story:

            return {
                "success": False,
                "error": "Please enter a story idea."
            }, 400


        prompt = f"""
Create a short comic story based on this idea:

{story}

IMPORTANT:
Create EXACTLY 4 comic panels.

Use this exact format:

Panel 1:
Scene: [scene description]
Dialogue: [dialogue]

Panel 2:
Scene: [scene description]
Dialogue: [dialogue]

Panel 3:
Scene: [scene description]
Dialogue: [dialogue]

Panel 4:
Scene: [scene description]
Dialogue: [dialogue]

Do not change the panel labels.
Do not add extra panels.

Keep the story creative, simple and suitable
for a college AI comic project.
"""


        response = None


        # -------------------------------------------------
        # TRY GEMINI UP TO 3 TIMES
        # -------------------------------------------------

        for attempt in range(3):

            try:

                response = gemini_client.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=prompt
                )

                break

            except Exception as error:

                print(
                    f"Gemini attempt {attempt + 1} failed:"
                )

                print(
                    str(error)
                )

                if (
                    "503" in str(error)
                    and attempt < 2
                ):

                    print(
                        "Retrying Gemini..."
                    )

                    time.sleep(3)

                else:

                    raise error


        if (
            response is None
            or not response.text
        ):

            raise RuntimeError(
                "Gemini did not return a story."
            )


        print("")
        print(
            "========== STORY GENERATED =========="
        )

        print(
            response.text
        )

        print(
            "======================================"
        )


        return {
            "success": True,
            "comic": response.text
        }


    except Exception as error:

        print("")
        print(
            "========== STORY GENERATION ERROR =========="
        )

        print(
            str(error)
        )

        print(
            "============================================="
        )


        app.logger.exception(
            "Comic story generation failed"
        )


        return {
            "success": False,
            "error": str(error)
        }, 500


# =========================================================
# GENERATE FOUR COMIC PANEL IMAGES
# USING HUGGING FACE
# =========================================================

@app.route(
    "/generate-images",
    methods=["POST"]
)
def generate_images():

    try:

        comic = request.json.get(
            "comic",
            ""
        )


        if not comic.strip():

            return {
                "success": False,
                "error": "Comic story is missing."
            }, 400


        # -------------------------------------------------
        # FIND THE FOUR PANELS
        # -------------------------------------------------

        panel_pattern = re.compile(
            r"(?:^|\n)\s*(?:#+\s*)?"
            r"Panel\s*([1-4])\s*[:.)-]?\s*",
            re.IGNORECASE
        )


        matches = list(
            panel_pattern.finditer(
                comic
            )
        )


        panels = []


        if len(matches) >= 4:

            for i in range(4):

                start = matches[i].end()


                if i < len(matches) - 1:

                    end = matches[i + 1].start()

                else:

                    end = len(comic)


                panel_text = comic[
                    start:end
                ].strip()


                panels.append(
                    panel_text
                )


        # -------------------------------------------------
        # FALLBACK PANEL DETECTION
        # -------------------------------------------------

        if len(panels) != 4:

            scene_parts = re.split(
                r"\n\s*(?=Scene\s*:)",
                comic,
                flags=re.IGNORECASE
            )


            scene_parts = [
                part.strip()
                for part in scene_parts
                if part.strip()
            ]


            if len(scene_parts) >= 4:

                panels = scene_parts[:4]


        if len(panels) != 4:

            return {
                "success": False,
                "error": (
                    "Could not identify all four "
                    "comic panels."
                )
            }, 400


        image_urls = []


        # =================================================
        # GENERATE EACH PANEL IMAGE
        # =================================================

        for index, panel in enumerate(
            panels,
            start=1
        ):

            # -------------------------------------------------
            # EXTRACT SCENE
            # -------------------------------------------------

            scene_match = re.search(
                r"Scene\s*:\s*(.*?)(?=Dialogue\s*:|$)",
                panel,
                flags=re.IGNORECASE | re.DOTALL
            )


            if scene_match:

                scene = (
                    scene_match
                    .group(1)
                    .strip()
                )

            else:

                scene = panel.strip()


            if not scene:

                scene = (
                    "A colorful college "
                    "classroom scene with "
                    "interesting characters."
                )


            # -------------------------------------------------
            # HUGGING FACE IMAGE PROMPT
            # -------------------------------------------------

            image_prompt = f"""
Create a beautiful colorful 2D cartoon
comic-book illustration.

Scene:
{scene}

Requirements:
- High-quality comic illustration
- Clear consistent characters
- Expressive emotions
- Bright attractive colors
- Detailed background
- Wide 4:3 composition
- Single comic panel
- No speech bubbles
- No captions
- No written text
- Clean outlines
- Suitable for a college AI project
"""


            print("")
            print(
                f"========== PANEL {index} =========="
            )

            print(
                "Connecting to Hugging Face..."
            )

            print(
                "Generating comic panel image..."
            )


            # -------------------------------------------------
            # GENERATE IMAGE USING FLUX
            # -------------------------------------------------

            image = hf_client.text_to_image(

                prompt=image_prompt,

                model=(
                    "black-forest-labs/"
                    "FLUX.1-schnell"
                ),
            )


            if image is None:

                raise RuntimeError(
                    f"Hugging Face returned "
                    f"no image for Panel {index}."
                )


            print(
                f"Panel {index} image generated!"
            )


            # -------------------------------------------------
            # SAVE IMAGE
            # -------------------------------------------------

            filename = (
                f"comic_panel_{index}.png"
            )


            filepath = os.path.join(
                IMAGE_FOLDER,
                filename
            )


            image.save(
                filepath
            )


            print(
                f"Panel {index} saved:"
            )

            print(
                filepath
            )


            # -------------------------------------------------
            # URL FOR FRONTEND
            # -------------------------------------------------

            image_urls.append(
                f"/static/generated/{filename}"
            )


        # =================================================
        # ALL IMAGES SUCCESSFULLY GENERATED
        # =================================================

        print("")
        print(
            "========== ALL 4 IMAGES GENERATED =========="
        )


        return {
            "success": True,
            "images": image_urls
        }


    except Exception as error:

        print("")
        print(
            "========== IMAGE GENERATION ERROR =========="
        )

        print(
            str(error)
        )

        print(
            "============================================="
        )


        app.logger.exception(
            "Comic image generation failed"
        )


        return {
            "success": False,
            "error": str(error)
        }, 500

        # =========================================================
# DOWNLOAD COMIC AS PDF
# =========================================================

@app.route(
    "/download-pdf",
    methods=["POST"]
)
def download_pdf():

    try:

        comic = request.json.get(
            "comic",
            ""
        )

        images = request.json.get(
            "images",
            []
        )

        if not comic.strip():

            return {
                "success": False,
                "error": "Comic story is missing."
            }, 400

        if len(images) != 4:

            return {
                "success": False,
                "error": "Four comic images are required."
            }, 400


        # -------------------------------------------------
        # PDF FILE PATH
        # -------------------------------------------------

        pdf_folder = os.path.join(
            "static",
            "generated"
        )

        os.makedirs(
            pdf_folder,
            exist_ok=True
        )

        pdf_path = os.path.join(
            pdf_folder,
            "ComicCraft_Comic.pdf"
        )


        # -------------------------------------------------
        # CREATE PDF
        # -------------------------------------------------

        pdf = canvas.Canvas(
            pdf_path,
            pagesize=A4
        )

        page_width, page_height = A4


        # -------------------------------------------------
        # TITLE PAGE
        # -------------------------------------------------

        pdf.setFont(
            "Helvetica-Bold",
            24
        )

        pdf.drawCentredString(
            page_width / 2,
            page_height - 60,
            "ComicCraft"
        )


        pdf.setFont(
            "Helvetica",
            12
        )

        pdf.drawCentredString(
            page_width / 2,
            page_height - 82,
            "AI Comic Story Creator"
        )


        # -------------------------------------------------
        # EXTRACT PANELS
        # -------------------------------------------------

        panel_pattern = re.compile(
            r"(?:^|\n)\s*(?:#+\s*)?"
            r"Panel\s*([1-4])\s*[:.)-]?\s*",
            re.IGNORECASE
        )

        matches = list(
            panel_pattern.finditer(
                comic
            )
        )

        panels = []

        if len(matches) >= 4:

            for i in range(4):

                start = matches[i].end()

                if i < 3:
                    end = matches[i + 1].start()
                else:
                    end = len(comic)

                panels.append(
                    comic[start:end].strip()
                )


        # -------------------------------------------------
        # ADD EACH PANEL
        # -------------------------------------------------

        for index in range(4):

            pdf.showPage()

            # Panel heading
            pdf.setFont(
                "Helvetica-Bold",
                20
            )

            pdf.drawString(
                50,
                page_height - 50,
                f"Panel {index + 1}"
            )


            # -------------------------------------------------
            # IMAGE
            # -------------------------------------------------

            image_path = os.path.join(
                "static",
                "generated",
                f"comic_panel_{index + 1}.png"
            )


            if os.path.exists(image_path):

                img = ImageReader(
                    image_path
                )

                image_width = 480
                image_height = 360

                pdf.drawImage(
                    img,
                    55,
                    page_height - 430,
                    width=image_width,
                    height=image_height,
                    preserveAspectRatio=True,
                    anchor="c"
                )


            # -------------------------------------------------
            # SCENE AND DIALOGUE
            # -------------------------------------------------

            panel_text = (
                panels[index]
                if index < len(panels)
                else ""
            )


            scene_match = re.search(
                r"Scene\s*:\s*(.*?)(?=Dialogue\s*:|$)",
                panel_text,
                flags=re.IGNORECASE | re.DOTALL
            )


            dialogue_match = re.search(
                r"Dialogue\s*:\s*(.*)",
                panel_text,
                flags=re.IGNORECASE | re.DOTALL
            )


            scene = (
                scene_match.group(1).strip()
                if scene_match
                else ""
            )


            dialogue = (
                dialogue_match.group(1).strip()
                if dialogue_match
                else ""
            )


            # -------------------------------------------------
            # TEXT
            # -------------------------------------------------

            y = page_height - 470

            pdf.setFont(
                "Helvetica-Bold",
                11
            )

            pdf.drawString(
                50,
                y,
                "Scene:"
            )

            y -= 18

            pdf.setFont(
                "Helvetica",
                10
            )


            # Wrap scene text
            scene_words = scene.split()
            line = ""

            for word in scene_words:

                test_line = (
                    line + " " + word
                ).strip()

                if pdf.stringWidth(
                    test_line,
                    "Helvetica",
                    10
                ) < 490:

                    line = test_line

                else:

                    pdf.drawString(
                        50,
                        y,
                        line
                    )

                    y -= 14
                    line = word

            if line:

                pdf.drawString(
                    50,
                    y,
                    line
                )

                y -= 25


            pdf.setFont(
                "Helvetica-Bold",
                11
            )

            pdf.drawString(
                50,
                y,
                "Dialogue:"
            )

            y -= 18

            pdf.setFont(
                "Helvetica",
                10
            )


            # Wrap dialogue text
            dialogue_words = dialogue.split()
            line = ""

            for word in dialogue_words:

                test_line = (
                    line + " " + word
                ).strip()

                if pdf.stringWidth(
                    test_line,
                    "Helvetica",
                    10
                ) < 490:

                    line = test_line

                else:

                    pdf.drawString(
                        50,
                        y,
                        line
                    )

                    y -= 14
                    line = word

            if line:

                pdf.drawString(
                    50,
                    y,
                    line
                )


        # -------------------------------------------------
        # SAVE PDF
        # -------------------------------------------------

        pdf.save()


        print("")
        print(
            "========== PDF CREATED =========="
        )

        print(
            pdf_path
        )


        return send_file(
            pdf_path,
            as_attachment=True,
            download_name="ComicCraft_Comic.pdf",
            mimetype="application/pdf"
        )


    except Exception as error:

        print("")
        print(
            "========== PDF ERROR =========="
        )

        print(
            str(error)
        )


        return {
            "success": False,
            "error": str(error)
        }, 500


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )