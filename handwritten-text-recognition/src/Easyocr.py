import easyocr
import cv2
import os

try:
    import Levenshtein
    USE_LEV = True
except:
    USE_LEV = False
    print("⚠️ Levenshtein not installed. Using exact match only.")

reader = easyocr.Reader(['en'])

data_path = "data/raw/"
label_file = "data/labels.txt"

correct = 0
total = 0
similarities = []

with open(label_file, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue

        try:
            filename, actual = line.strip().split(',', 1)
        except:
            print(f" Skipping bad line: {line}")
            continue

        img_path = os.path.join(data_path, filename)

        if not os.path.exists(img_path):
            print(f" Image not found: {img_path}")
            continue

        image = cv2.imread(img_path)

        if image is None:
            print(f" Failed to load image: {img_path}")
            continue

        # OCR
        result = reader.readtext(image)

        if len(result) == 0:
            predicted = ""
        else:
            predicted = " ".join([res[1] for res in result])

        # Normalize
        actual_clean = actual.lower().strip()
        predicted_clean = predicted.lower().strip()

        # Similarity (if available)
        if USE_LEV:
            similarity = Levenshtein.ratio(predicted_clean, actual_clean)
        else:
            similarity = 1.0 if predicted_clean == actual_clean else 0.0

        similarities.append(similarity)

        # Threshold-based correctness (IMPORTANT)
        if similarity >= 0.7:
            correct += 1

        total += 1

        print(f"Actual: {actual_clean} | Predicted: {predicted_clean} | Similarity: {similarity:.2f}")

# Final metrics
if total == 0:
    print(" No valid samples found. Check your paths/labels.")
else:
    accuracy = correct / total
    avg_similarity = sum(similarities) / len(similarities)

    print("\n========== RESULTS ==========")
    print(f"Total Samples: {total}")
    print(f"Accuracy (>=70% match): {accuracy:.2f}")
    print(f"Average Similarity: {avg_similarity:.2f}")