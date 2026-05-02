import easyocr
import cv2
import os
import csv
from preprocess import preprocess_image
from augmentation import augment_image
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

summary_file = os.path.join(BASE_DIR, "..", "results", "final_results.csv")
detailed_file = os.path.join(BASE_DIR, "..", "results", "detailed_results.csv")

# =========================
# SETTINGS
# =========================

aug_methods = [
    "none",
    "rotate",
    "noise",
    "scale",
    "rotate_noise"
]

methods = [
    "gray",
    "resize",
    "gray_resize",
    "blur",
    "threshold",
    "adaptive",
    "blur_threshold"
]

# =========================
# SETUP FILES
# =========================

os.makedirs(os.path.dirname(summary_file), exist_ok=True)

# Summary CSV
with open(summary_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        "augmentation",
        "preprocessing",
        "raw_accuracy",
        "corrected_accuracy",
        "raw_avg_similarity",
        "corrected_avg_similarity"
    ])

# Detailed CSV
with open(detailed_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        "augmentation",
        "preprocessing",
        "filename",
        "actual",
        "raw_output",
        "corrected_output",
        "raw_similarity",
        "corrected_similarity"
    ])

# =========================
# MAIN PIPELINE
# =========================

for aug in aug_methods:
    print(f"\n==============================")
    print(f"AUGMENTATION: {aug}")
    print(f"==============================")

    for method in methods:
        print(f"\n--- PREPROCESS: {method} ---")

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
                    print(f"❌ Could not load: {filename}")
                    continue

                # STEP 1: AUGMENTATION
                if aug != "none":
                    image = augment_image(image, aug)

                # STEP 2: PREPROCESSING
                processed = preprocess_image(image, method)

                # STEP 3: OCR
                result = reader.readtext(processed)
                raw_text = " ".join([r[1] for r in result]) if result else ""

                # STEP 4: CONTEXT CORRECTION
                corrected_text = correct_text(raw_text)

                # STEP 5: NORMALIZATION
                actual_clean = actual.lower().strip()
                raw_clean = raw_text.lower().strip()
                corrected_clean = corrected_text.lower().strip()

                # STEP 6: SIMILARITY
                if USE_LEV:
                    raw_similarity = Levenshtein.ratio(raw_clean, actual_clean)
                    corrected_similarity = Levenshtein.ratio(corrected_clean, actual_clean)
                else:
                    raw_similarity = 1.0 if raw_clean == actual_clean else 0.0
                    corrected_similarity = 1.0 if corrected_clean == actual_clean else 0.0

                raw_similarities.append(raw_similarity)
                corrected_similarities.append(corrected_similarity)

                # STEP 7: ACCURACY COUNT
                if raw_similarity >= 0.7:
                    raw_correct += 1

                if corrected_similarity >= 0.7:
                    corrected_correct += 1

                total += 1

                # =========================
                # PRINT ONLY IMPORTANT CASES
                # =========================
                if raw_clean != corrected_clean:
                    print("\n--- Correction Example ---")
                    print(f"Image      : {filename}")
                    print(f"Actual     : {actual_clean}")
                    print(f"Raw OCR    : {raw_clean}")
                    print(f"Corrected  : {corrected_clean}")

                # =========================
                # SAVE DETAILED RESULT
                # =========================
                with open(detailed_file, mode='a', newline='', encoding='utf-8') as df:
                    writer = csv.writer(df)
                    writer.writerow([
                        aug,
                        method,
                        filename,
                        actual_clean,
                        raw_clean,
                        corrected_clean,
                        round(raw_similarity, 4),
                        round(corrected_similarity, 4)
                    ])

        # =========================
        # FINAL METRICS
        # =========================
        if total > 0:
            raw_accuracy = raw_correct / total
            corrected_accuracy = corrected_correct / total

            raw_avg_similarity = sum(raw_similarities) / len(raw_similarities)
            corrected_avg_similarity = sum(corrected_similarities) / len(corrected_similarities)

            print(f"\nRAW Accuracy: {raw_accuracy:.2f}")
            print(f"CORRECTED Accuracy: {corrected_accuracy:.2f}")

            # =========================
            # SAVE SUMMARY
            # =========================
            with open(summary_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    aug,
                    method,
                    round(raw_accuracy, 4),
                    round(corrected_accuracy, 4),
                    round(raw_avg_similarity, 4),
                    round(corrected_avg_similarity, 4)
                ])

print("\n✅ DONE — Results saved in 'results/' folder")