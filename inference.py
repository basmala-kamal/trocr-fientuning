from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

processor = TrOCRProcessor.from_pretrained("./trocr-captcha-finetuned")
model = VisionEncoderDecoderModel.from_pretrained("./trocr-captcha-finetuned")


img = Image.open("Search_By_BL_No (2).png").convert("RGB")
pixel_values = processor(img, return_tensors="pt").pixel_values
with torch.no_grad():
    generated_ids = model.generate(pixel_values, max_length=12)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(text)
