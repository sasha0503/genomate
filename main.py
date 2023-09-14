import re
import os
import time
import base64
import traceback

import uvicorn
import openai
import dotenv
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, HTTPException
from starlette.middleware.cors import CORSMiddleware

from generator_communicator import GeneratorCommunicator

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
with open('prompt.txt', 'r') as f:
    gpt_pre_prompt = f.read()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=True,
    max_age=600,
)


class ImageDB:
    def __init__(self, db_path):
        self.db_path = db_path

    def get_image(self, image_id):
        image_path = os.path.join(self.db_path, image_id) + '.png'
        if not os.path.exists(image_path):
            image_path = os.path.join(self.db_path, image_id) + '.jpg'
            if not os.path.exists(image_path):
                raise HTTPException(status_code=500, detail="Image not found")
        return Image.open(image_path)

    def save_image(self, image: Image):
        if not os.path.exists(self.db_path):
            os.mkdir(self.db_path)
            image_id = '0'
        else:
            ids = [f.split('.')[0] for f in os.listdir(self.db_path)]
            if len(ids) == 0:
                image_id = '0'
            else:
                ids.sort()
                last_id = int(ids[-1])
                image_id = str(last_id + 1)

        image_path = os.path.join(self.db_path, image_id) + '.png'
        if os.path.exists(image_path):
            # TODO better exceptions. Discuss with Igor
            raise HTTPException(status_code=500, detail="Image already exists")
        image.save(image_path, format="PNG")
        return image_id


class ServerPipe:
    def __init__(self, custom_generator: GeneratorCommunicator, images_database: ImageDB):
        self.generator: GeneratorCommunicator = custom_generator
        self.image_db: ImageDB = images_database

    def generate(self, prompt: str = "", from_scratch: bool = True, example_id: str = None) -> Image:
        if from_scratch and not example_id:
            res = self.generator.generate(prompt)
            img_id = self.image_db.save_image(res)
            return res, img_id

        else:
            example_image = self.image_db.get_image(example_id)
            res = self.generator.generate(prompt, example_image)
            img_id = self.image_db.save_image(res)
            return res, img_id


is_running = False
generator = GeneratorCommunicator(port=8080)
image_db = ImageDB('images_db')
server_pipe = ServerPipe(generator, image_db)


@app.post("/image-from-prompt/")
async def generate_image_from_prompt(prompt: str = "", from_scratch: bool = True, example_img_id: str = None):
    try:
        start = time.time()
        global is_running
        is_running = True

        image, new_img_id = server_pipe.generate(prompt, from_scratch, example_img_id)
        print("Generated image with id: " + new_img_id)
        print("Time to generate the image: " + str(time.time() - start))

        # Convert the image to bytes
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        is_running = False
        print("Time to send the image: " + str(time.time() - start))
        return {
            "id": new_img_id,
            "base64": image_base64,
        }
    except Exception as e:
        is_running = False
        traceback.print_exc()
        raise HTTPException(status_code=500)


@app.get("/create-script")
async def create_script(brief: str):
    try:
        # ---------- OpenAI API ----------
        conversation = [
            {"role": "system", "content": gpt_pre_prompt},
            {"role": "user", "content": brief}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation
        )

        # ---------- Getting result ----------
        text = response.choices[0]['message']['content']
        print(text)
        lines = text.split('\n')
        lines = [line for line in lines if line != '']
        lines = [re.split(r'\d+:', line)[-1] for line in lines]

        return lines
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500)


@app.get("/test")
async def test():
    try:
        return "Hello World"
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050)
