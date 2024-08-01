from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import requests
from PIL import Image

class Recognizer:
    def __init__(self):
        self.processor = TrOCRProcessor.from_pretrained("recognizer/")
        self.model = VisionEncoderDecoderModel.from_pretrained("recognizer/")

    def recognize(self, image):
        pixel_values = self.processor.__call__(image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
