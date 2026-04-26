import pytesseract
import cv2
import os

data_path = "data/raw/"
label_file = "data/labels.txt"

correct = 0
total = 0

for line in open(label_file):
    filename, actual = line.strip().split(",", 1)

    img_path = os.path.join(data_path, filename)
    image = cv2.imread(img_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    predicted = pytesseract.image_to_string(gray)

    print(f"Actual: {actual} | Predicted: {predicted.strip()}")

    if actual.lower() in predicted.lower():
        correct += 1

    total += 1

print("Accuracy:", correct / total)