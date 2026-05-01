import easyocr
import cv2
import os
import csv
from preprocess import preprocess_image
from context_correction import correct_text

try:
    import Levenshtein
    USE_LEV = True
except:
    USE_LEV = False

reader = easyocr.Reader(['en'])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(BASE_DIR, "..", "data", "raw")
label_file = os.path.join(BASE_DIR, "..", "data", "labels.txt")

# 🔥 RESULTS FILE (CSV)
results_file = os.path.join(BASE_DIR, "..", "results", "summary_results.csv")

methods = [
    "gray",
    "resize",
    "gray_resize",
    "blur",
    "threshold",
    "adaptive",
    "blur_threshold"
]

# 🔥 Ensure results folder exists
os.makedirs(os.path.dirname(results_file), exist_ok=True)

# 🔥 Create / overwrite CSV once
with open(results_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        "method",
        "raw_accuracy",
        "corrected_accuracy",
        "raw_avg_similarity",
        "corrected_avg_similarity"
    ])

# =========================
# MAIN LOOP
# =========================
for method in methods:
    print(f"\n===== METHOD: {method} =====")

    raw_correct = 0
    corrected_correct = 0
    total = 0

    raw_similarities = []
    corrected_similarities = []

    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            filename, actual = line.strip().split(',', 1)
            img_path = os.path.join(data_path, filename)

            image = cv2.imread(img_path)
            if image is None:
                continue

            # 🔥 PREPROCESSING
            processed = preprocess_image(image, method)

            # 🔥 OCR
            result = reader.readtext(processed)

            raw_text = " ".join([r[1] for r in result]) if result else ""

            # 🔥 CONTEXT CORRECTION
            corrected_text = correct_text(raw_text)

            # 🔥 NORMALIZE
            actual_clean = actual.lower().strip()
            raw_clean = raw_text.lower().strip()
            corrected_clean = corrected_text.lower().strip()

            # 🔥 SIMILARITY
            if USE_LEV:
                raw_similarity = Levenshtein.ratio(raw_clean, actual_clean)
                corrected_similarity = Levenshtein.ratio(corrected_clean, actual_clean)
            else:
                raw_similarity = 1.0 if raw_clean == actual_clean else 0.0
                corrected_similarity = 1.0 if corrected_clean == actual_clean else 0.0

            raw_similarities.append(raw_similarity)
            corrected_similarities.append(corrected_similarity)

            # 🔥 ACCURACY
            if raw_similarity >= 0.7:
                raw_correct += 1

            if corrected_similarity >= 0.7:
                corrected_correct += 1

            total += 1

            # 🔥 DEBUG EXAMPLE
            if raw_clean != corrected_clean:
                print("\n--- Example ---")
                print("Actual    :", actual_clean)
                print("Raw OCR   :", raw_clean)
                print("Corrected :", corrected_clean)

    # =========================
    # METRICS CALCULATION
    # =========================
    if total > 0:
        raw_accuracy = raw_correct / total
        corrected_accuracy = corrected_correct / total

        raw_avg_similarity = sum(raw_similarities) / len(raw_similarities)
        corrected_avg_similarity = sum(corrected_similarities) / len(corrected_similarities)

        print(f"\nRAW Accuracy: {raw_accuracy:.2f}")
        print(f"RAW Avg Similarity: {raw_avg_similarity:.2f}")

        print(f"CORRECTED Accuracy: {corrected_accuracy:.2f}")
        print(f"CORRECTED Avg Similarity: {corrected_avg_similarity:.2f}")

        # =========================
        # SAVE TO CSV
        # =========================
        with open(results_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                method,
                round(raw_accuracy, 4),
                round(corrected_accuracy, 4),
                round(raw_avg_similarity, 4),
                round(corrected_avg_similarity, 4)
            ])