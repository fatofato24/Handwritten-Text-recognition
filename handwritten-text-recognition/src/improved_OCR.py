import easyocr
import cv2
import os
import csv
import Levenshtein

from preprocess import preprocess_image
from context_correction import correct_text

# =========================
# OCR ENGINE
# =========================
reader = easyocr.Reader(['en'])


# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(BASE_DIR, "..", "data", "raw")
label_file = os.path.join(BASE_DIR, "..", "data", "labels.txt")

summary_file = os.path.join(BASE_DIR, "..", "results", "summary_results.csv")
detailed_file = os.path.join(BASE_DIR, "..", "results", "detailed_results.csv")

os.makedirs(os.path.dirname(summary_file), exist_ok=True)


# =========================
# METHODS
# =========================
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
# CSV INIT
# =========================
with open(summary_file, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow([
        "method",
        "raw_accuracy",
        "corrected_accuracy",
        "raw_avg_similarity",
        "corrected_avg_similarity"
    ])

with open(detailed_file, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow([
        "method","filename","actual","raw","corrected",
        "raw_similarity","corrected_similarity"
    ])


# =========================
# MAIN LOOP
# =========================
for method in methods:
    print(f"\n===== METHOD: {method} =====")

    raw_correct = 0
    corrected_correct = 0
    total = 0

    raw_sim_list = []
    corr_sim_list = []

    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            filename, actual = line.strip().split(",", 1)
            img_path = os.path.join(data_path, filename)

            image = cv2.imread(img_path)
            if image is None:
                continue

            # =========================
            # PREPROCESS
            # =========================
            processed = preprocess_image(image, method)

            # =========================
            # OCR
            # =========================
            result = reader.readtext(processed)

            raw_text = " ".join([r[1] for r in result]) if result else ""

            # =========================
            # ENHANCED CONTEXT CORRECTION (with ensemble voting)
            # =========================
            corrected_text = correct_text(raw_text, result)

            # =========================
            # NORMALIZE
            # =========================
            actual_c = actual.lower().strip()
            raw_c = raw_text.lower().strip()
            corr_c = corrected_text.lower().strip()

            # =========================
            # SIMILARITY
            # =========================
            raw_sim = Levenshtein.ratio(raw_c, actual_c)
            corr_sim = Levenshtein.ratio(corr_c, actual_c)

            raw_sim_list.append(raw_sim)
            corr_sim_list.append(corr_sim)

            # =========================
            # ACCURACY
            # =========================
            if raw_sim >= 0.7:
                raw_correct += 1
            if corr_sim >= 0.7:
                corrected_correct += 1

            total += 1

            # =========================
            # DEBUG
            # =========================
            if raw_c != corr_c:
                print("\n--- Example ---")
                print("Actual    :", actual_c)
                print("Raw OCR   :", raw_c)
                print("Corrected :", corr_c)

            # =========================
            # SAVE DETAIL
            # =========================
            with open(detailed_file, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    method, filename, actual_c,
                    raw_c, corr_c,
                    round(raw_sim,4),
                    round(corr_sim,4)
                ])

    # =========================
    # FINAL RESULTS
    # =========================
    if total > 0:
        print(f"\nRAW Accuracy: {raw_correct/total:.2f}")
        print(f"CORRECTED Accuracy: {corrected_correct/total:.2f}")
        print(f"RAW Avg Similarity: {sum(raw_sim_list)/total:.2f}")
        print(f"CORR Avg Similarity: {sum(corr_sim_list)/total:.2f}")

        with open(summary_file, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                method,
                round(raw_correct/total,4),
                round(corrected_correct/total,4),
                round(sum(raw_sim_list)/total,4),
                round(sum(corr_sim_list)/total,4)
            ])