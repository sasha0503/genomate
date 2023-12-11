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

"""
ключ для доступу до openai api зберігається у файлі .env який не включений до репозиторію з міркувань безпеки
"""
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
        """
        Отримує зображення з бази даних (папки) по id
        """
        image_path = os.path.join(self.db_path, image_id) + '.png'
        if not os.path.exists(image_path):
            image_path = os.path.join(self.db_path, image_id) + '.jpg'
            if not os.path.exists(image_path):
                raise HTTPException(status_code=500, detail="Image not found")
        return Image.open(image_path)

    def save_image(self, image: Image):
        """
        Зберігає зображення в базу даних (папку) і повертає його id
         id - це порядковий номер зображення в базі даних
        """
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
    """
    Клас ServerPipe - це клас, який зберігає в собі об'єкти класів GeneratorCommunicator та ImageDB.
    Його метод generate приймає prompt, example_id та from_scratch і повертає зображення та його id.

    Клас ServerPipe використовується для зручної взаємодії з генератором та базою даних зображень в одному місці.
    """
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
async def generate_image_from_prompt(prompts):
    """
    :param prompts: list of dict with keys: description: str, scene_id: str, from_scratch: bool, example_img_id: str
    """
    try:
        start = time.time()
        """
        використовується параметр is_running, щоб уникнути одночасного запуску багатьох процесів генерації
        """
        global is_running
        is_running = True

        byte64s = []
        ids = []
        scene_ids = []
        for prompt in prompts:
            """
            prompt - це словник з ключами description, scene_id, from_scratch, example_img_id
            є можливість обробляти декілька prompt в одному запиті (при цьому вони будуть оброблятись послідовно)
            """
            description = prompt.get('description', '')
            scene_id = prompt.get('scene_id', '')
            from_scratch = prompt.get('from_scratch', True)
            example_img_id = prompt.get('example_img_id', None)
            if not description or not scene_id:
                raise HTTPException(status_code=500, detail="Invalid prompt")

            image, new_img_id = server_pipe.generate(description, from_scratch, example_img_id)
            print("Generated image with id: " + new_img_id)
            print("Time to generate the image: " + str(time.time() - start))
            ids.append(new_img_id)
            scene_ids.append(scene_id)

            # Convert the image to bytes
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            byte64s.append(image_base64)

        is_running = False
        print("Time to send the image: " + str(time.time() - start))

        """
        повертаємо послідовність зображень у форматі base64, їх id та scene_id для кожного prompt із запиту
        """
        return [{"img64": byte64, "id": id, "scene_id": scene_id} for byte64, id, scene_id in zip(byte64s, ids, scene_ids)]
    except Exception as e:
        is_running = False
        traceback.print_exc()
        raise HTTPException(status_code=500)


@app.get("/create-script")
async def create_script(brief: str):
    try:
        """
        генерація скрипту за допомогою openai api
        """
        # ---------- OpenAI API ----------
        if brief == "":
            raise HTTPException(status_code=500, detail="Invalid prompt")
        """
        для кастомізації генерації можна використовувати role та content
        у даному випадку використовується role: system, content: gpt_pre_prompt - опис скрипту, який ми хочемо отримати
        role: user, content: brief - опис сцени від юзера, який ми передаємо у запиті
        """
        conversation = [
            {"role": "system", "content": gpt_pre_prompt},
            {"role": "user", "content": brief}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation
        )

        # ---------- Getting result ----------
        """
        отримання та обробка результату генерації
        """
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
