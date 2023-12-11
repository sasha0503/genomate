import random
import base64
import io

import uvicorn
from PIL import Image
from fastapi import FastAPI

try:
    import torch
    from torch.nn import DataParallel
    from diffusers import DiffusionPipeline
except ImportError:
    print('Diffusion models are not available')


class Generator:
    """
    Клас Generator - це абстрактний клас, який містить метод __call__.
    """
    def __call__(self, prompt: str, example_img: Image.Image = None) -> Image:
        raise NotImplementedError


class DummyGenerator(Generator):
    """
    Клас DummyGenerator - це клас-нащадок класу Generator, який просто повертає зображення з диску.
    Використовуваєтсья для тестування.
    """
    def __call__(self, prompt, example_img=None) -> Image:
        if example_img is None:
            image = Image.open('astronaut_on_horse.png')
            return image
        else:
            width, height = example_img.size
            random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), int(255 * 0.3))
            overlay = Image.new('RGBA', (width, height), random_color)
            result = Image.alpha_composite(example_img.convert('RGBA'), overlay)
        return result


class StableDiffusionGenerator(Generator):
    """
    Клас StableDiffusionGenerator - це клас-нащадок класу Generator, який використовує модель Stable Diffusion для генерації зображень.
    При ініціалізації класу завантажується модель з хабу, яка зберігається в полі pipe.

    При виклику класу викликається метод __call__, який приймає prompt - текст, який буде використовуватись для генерації зображення.
    """
    def __init__(self):
        self.pipe = DiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to("cuda:0")

    def __call__(self, prompt, example_img=None) -> Image:
        res = self.pipe(prompt).images[0]
        return res


if __name__ == '__main__':
    """
    Для комунікації з генератором використовується FastAPI та реалізовано один ендпоінт /generate, який приймає POST-запити
    і викликає метод __call__ класу Generator і повертає зображення у форматі base64.
    """
    app = FastAPI()
    generator = StableDiffusionGenerator()


    @app.post("/generate/")
    def generate(prompt: str = "", example_img=None):
        pil_img = generator(prompt, example_img)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"img64": image_base64}


    uvicorn.run(app, host="0.0.0.0", port=8080)
