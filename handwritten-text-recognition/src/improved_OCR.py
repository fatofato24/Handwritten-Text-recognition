import easyocr
import cv2
import os
from preprocess import preprocess_image

try:
    import Levenshtein
    USE_LEV = True
except:
    USE_LEV = False

reader = easyocr.Reader(['en'])

data_path = "data/raw/"
label_file = "data/labels.txt"

methods = [
    "gray",
    "resize",
    "gray_resize",
    "blur",
    "threshold",
    "adaptive",
    "blur_threshold"
]

for method in methods:
    print(f"\n===== METHOD: {method} =====")

    correct = 0
    total = 0
    similarities = []

    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            filename, actual = line.strip().split(',', 1)
            img_path = os.path.join(data_path, filename)

            image = cv2.imread(img_path)
            if image is None:
                continue

            # 🔥 APPLY PREPROCESSING
            processed = preprocess_image(image, method)

            # OCR
            result = reader.readtext(processed)

            predicted = ""
            if len(result) > 0:
                predicted = " ".join([r[1] for r in result])

            # normalize
            actual_clean = actual.lower().strip()
            predicted_clean = predicted.lower().strip()

            # similarity
            if USE_LEV:
                similarity = Levenshtein.ratio(predicted_clean, actual_clean)
            else:
                similarity = 1.0 if predicted_clean == actual_clean else 0.0

            similarities.append(similarity)

            if similarity >= 0.7:
                correct += 1

            total += 1

    if total > 0:
        accuracy = correct / total
        avg_similarity = sum(similarities) / len(similarities)

        print(f"Accuracy: {accuracy:.2f}")
        print(f"Avg Similarity: {avg_similarity:.2f}")