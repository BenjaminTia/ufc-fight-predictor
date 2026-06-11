"""
Environment Check Script
Verifies CUDA, PyTorch, XGBoost, and LightGBM GPU acceleration are working.
Run: python scripts/check_environment.py
"""

import sys
import platform

def check_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def check_success(msg):
    print(f"  [OK] {msg}")

def check_fail(msg):
    print(f"  [FAIL] {msg}")

check_header("System Information")
print(f"  OS: {platform.system()} {platform.release()}")
print(f"  Python: {sys.version.split()[0]}")
print(f"  Architecture: {platform.machine()}")

check_header("PyTorch + CUDA")
try:
    import torch
    check_success(f"PyTorch {torch.__version__} imported")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        check_success(f"CUDA Available: {cuda_available}")
        check_success(f"GPU: {torch.cuda.get_device_name(0)}")
        check_success(f"CUDA Version: {torch.version.cuda}")
        check_success(f"GPU Count: {torch.cuda.device_count()}")
        check_success(f"Current Device: {torch.cuda.current_device()}")

        # Test tensor creation on GPU
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        check_success(f"GPU matmul test: {z.shape} computed on {z.device}")
        del x, y, z
        torch.cuda.empty_cache()
    else:
        check_fail("CUDA is NOT available. Check NVIDIA driver and CUDA toolkit installation.")
except ImportError:
    check_fail("PyTorch not installed. Run: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

check_header("XGBoost GPU")
try:
    import xgboost as xgb
    check_success(f"XGBoost {xgb.__version__} imported")
    import numpy as np
    try:
        X = np.random.randn(500, 20)
        y = np.random.randint(0, 2, 500)
        model = xgb.XGBClassifier(
            tree_method='hist', device='cuda', n_estimators=10,
            max_depth=3, verbosity=0
        )
        model.fit(X, y)
        preds = model.predict_proba(X)
        check_success(f"XGBoost GPU training OK. Predict shape: {preds.shape}")
    except Exception as e:
        check_fail(f"XGBoost GPU failed: {e}")
        print("  Fallback: try tree_method='hist', device='cpu' in model_training.py")
except ImportError:
    check_fail("XGBoost not installed. Run: pip install xgboost==2.1.1")

check_header("LightGBM GPU")
try:
    import lightgbm as lgb
    check_success(f"LightGBM {lgb.__version__} imported")
    try:
        X = np.random.randn(500, 20)
        y = np.random.randint(0, 2, 500)
        model = lgb.LGBMClassifier(
            device_type='gpu', n_estimators=10, max_depth=3,
            verbose=-1
        )
        model.fit(X, y)
        preds = model.predict_proba(X)
        check_success(f"LightGBM GPU training OK. Predict shape: {preds.shape}")
    except Exception as e:
        check_fail(f"LightGBM GPU failed: {e}")
        print("  Tip: On Windows, LightGBM GPU requires Visual Studio Build Tools with C++ workload.")
        print("  Fallback: use device_type='cpu' in model_training.py")
except ImportError:
    check_fail("LightGBM not installed. Run: pip install lightgbm==4.5.0")

check_header("Transformers (NLP)")
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    check_success("Transformers imported successfully")
    try:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        if cuda_available:
            model = model.to("cuda")
            check_success("NLP model moved to GPU successfully")
    except Exception as e:
        print(f"  [WARN] Could not load test NLP model: {e}")
        print("  This is fine -- models download on first use in scrape_news_sentiment.py")
except ImportError:
    check_fail("Transformers not installed. Run: pip install transformers")

check_header("Scraping Libraries")
for lib in ["requests", "bs4", "pandas", "sklearn", "matplotlib", "seaborn", "shap", "joblib", "tqdm"]:
    try:
        __import__(lib)
        check_success(f"{lib} OK")
    except ImportError:
        check_fail(f"{lib} not installed")

check_header("Summary")
if cuda_available:
    print("  SUCCESS: Environment is ready for GPU-accelerated training!")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("  WARNING: CUDA not available. Training will run on CPU (slow).")
    print("  Install CUDA Toolkit 12.1 and update NVIDIA drivers.")

print("\n  Next steps:")
print("    1. python scripts/scrape_ufcstats.py")
print("    2. python scripts/scrape_expert_predictions.py")
print("    3. python scripts/scrape_news_sentiment.py")
print("    4. python scripts/feature_engineering.py")
print("    5. python scripts/model_training.py")
print("    6. python scripts/predict_fight.py")
print()
