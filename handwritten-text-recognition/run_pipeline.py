"""
Master Pipeline Runner for EasyOCR Evaluation
Orchestrates the complete baseline evaluation workflow.

Usage:
    python run_pipeline.py

This script:
1. Runs EasyOCR inference on all images
2. Saves predictions to CSV
3. Evaluates predictions against ground truth
4. Generates comprehensive metrics and reports
"""

import subprocess
import sys
import os

def run_script(script_name: str, description: str) -> bool:
    """Run a Python script and return success status."""
    print(f"\n{'='*70}")
    print(f"▶️  {description}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=os.path.dirname(os.path.abspath(__file__)) or '.',
            capture_output=False
        )
        
        if result.returncode == 0:
            print(f"\n✅ {description} - SUCCESS")
            return True
        else:
            print(f"\n❌ {description} - FAILED")
            return False
    except Exception as e:
        print(f"\n❌ Error running {script_name}: {e}")
        return False


def main():
    """Run the complete evaluation pipeline."""
    print("\n" + "="*70)
    print("🚀 HANDWRITTEN TEXT RECOGNITION - EASYOCR BASELINE PIPELINE")
    print("="*70)
    
    scripts = [
        ("src/Easyocr.py", "Step 1: EasyOCR Inference - Generate predictions"),
        ("src/evaluate.py", "Step 2: Evaluation - Compute metrics and generate reports"),
    ]
    
    successful = 0
    failed = 0
    
    for script, description in scripts:
        if run_script(script, description):
            successful += 1
        else:
            failed += 1
    
    # Final summary
    print(f"\n{'='*70}")
    print("📊 PIPELINE SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Successful: {successful}/{len(scripts)}")
    print(f"❌ Failed: {failed}/{len(scripts)}")
    
    if failed == 0:
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"\n📁 Generated files:")
        print(f"   📊 Predictions: results/easyocr_predictions.csv")
        print(f"   📈 Evaluation:  results/easyocr_evaluation.csv")
        print(f"   📋 Summary:     results/easyocr_summary.txt")
        print(f"\n💡 Next steps:")
        print(f"   1. Review results/easyocr_summary.txt for key insights")
        print(f"   2. Run Tesseract and TrOCR baselines")
        print(f"   3. Compare results across all three OCR engines")
    else:
        print(f"\n⚠️  Pipeline encountered errors. Please check the output above.")
        return 1
    
    print("="*70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
