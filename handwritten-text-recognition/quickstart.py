#!/usr/bin/env python
"""
🚀 QUICK START - Run this to test the improvements immediately!
Execute: python quickstart.py
"""

import os
import subprocess
import sys

def run_command(cmd, description):
    """Run a command and show status"""
    print(f"\n{'='*70}")
    print(f"📍 {description}")
    print(f"{'='*70}")
    print(f"Running: {cmd}\n")
    
    result = os.system(cmd)
    
    if result == 0:
        print(f"\n✅ {description} - SUCCESS")
    else:
        print(f"\n❌ {description} - FAILED (Exit code: {result})")
        return False
    
    return True


def main():
    print(f"\n{'='*70}")
    print(f"🚀 OCR IMPROVEMENT QUICKSTART")
    print(f"{'='*70}")
    
    # Ensure we're in the right directory
    if not os.path.exists("data/labels.txt"):
        print("❌ Please run this from the 'handwritten-text-recognition' directory")
        print("   cd handwritten-text-recognition")
        sys.exit(1)
    
    # Step 1: Check dependencies
    print(f"\n{'='*70}")
    print("📋 STEP 1: Checking Dependencies")
    print(f"{'='*70}")
    
    try:
        import easyocr
        import cv2
        import torch
        from PIL import Image
        from transformers import TrOCRProcessor
        from symspellpy import SymSpell
        import Levenshtein
        import wordfreq
        print("✅ All dependencies found!")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print(f"   Run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Step 2: Run EasyOCR baseline
    success = run_command(
        "python src/Easyocr.py",
        "BASELINE - Running EasyOCR on all images"
    )
    if not success:
        sys.exit(1)
    
    # Step 3: Run Ensemble
    success = run_command(
        "python src/ensemble_ocr.py",
        "ENSEMBLE - Running multi-engine OCR with voting"
    )
    if not success:
        sys.exit(1)
    
    # Step 4: Show comparison
    success = run_command(
        "python src/compare_results.py",
        "VISUALIZATION - Showing improvements"
    )
    if not success:
        sys.exit(1)
    
    # Step 5: Detailed evaluation
    success = run_command(
        "python src/evaluate_ensemble.py",
        "EVALUATION - Detailed metrics and analysis"
    )
    if not success:
        sys.exit(1)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"✅ QUICKSTART COMPLETE!")
    print(f"{'='*70}")
    print(f"""
📊 Results saved to:
   • results/easyocr_predictions.csv         (Baseline)
   • results/ensemble_predictions.csv         (Ensemble improved)
   • results/ensemble_comparison.csv          (All engines comparison)
   • results/analysis_results.csv             (Detailed metrics)

📈 Check Results:
   1. Look at the comparison table above
   2. Review results/ensemble_comparison.csv in Excel
   3. Check metrics in results/analysis_results.csv

🎯 Next Steps:
   1. If accuracy improved → Use ensemble in production!
   2. If accuracy didn't improve → Fine-tune parameters:
      • Adjust confidence thresholds in src/context_correction.py
      • Modify voting weights in src/ensemble_ocr.py
      • Add more OCR error patterns

📚 For more details:
   • Read: ../IMPROVEMENTS_GUIDE.md
   • Explore: src/context_correction.py (main improvement logic)
   • Experiment: src/ensemble_ocr.py (voting strategy)

Happy improving! 🚀
    """)


if __name__ == "__main__":
    main()
