import base64
import io

import requests
from PIL import Image


class GeneratorCommunicator:
    def __init__(self, port, host='localhost'):
        self.port = port
        self.host = host

    def generate(self, prompt, example_img=None):
        req = requests.post(f'http://{self.host}:{self.port}/generate',
                            json={'prompt': prompt, 'example_img': example_img})
        if req.status_code != 200:
            raise Exception(f'Generator error: {req.text}')
        base64_image_string = req.json()['img64']
        image_bytes = base64.b64decode(base64_image_string)
        image_buffer = io.BytesIO(image_bytes)
        image = Image.open(image_buffer)
        return image


if __name__ == '__main__':
    communicator = GeneratorCommunicator(8080)
    res = communicator.generate('test')
    print(res)
