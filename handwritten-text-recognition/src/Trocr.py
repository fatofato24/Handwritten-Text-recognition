from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import os

print(" Starting TrOCR test...")

# Load model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

print(" Model loaded")

# Test image path
img_path = "data/raw/image (1).jpeg"

if not os.path.exists(img_path):
    print(" Image not found:", img_path)
    exit()

# Load image
image = Image.open(img_path).convert("RGB")
print(" Image loaded")

# Run OCR
pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)

predicted = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(" Prediction:", predicted)