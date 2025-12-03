"""
Setup script to create directory structure and verify dependencies.
Run this first: python setup.py
"""

from pathlib import Path
import sys


def create_directories():
    """Create necessary directory structure"""
    base_path = Path(__file__).parent
    
    directories = [
        "data/raw/tweets",
        "data/raw/markets",
        "data/processed/labeled_tweets",
        "data/processed/features",
        "data/models",
        "src/data_collection",
        "src/preprocessing",
        "src/labeling",
        "src/model",
        "src/prediction",
        "src/evaluation",
        "notebooks",
        "config",
    ]
    
    for dir_path in directories:
        full_path = base_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {full_path}")


def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'pandas',
        'numpy',
        'yaml',
        'tqdm',
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"\nWarning: Missing packages: {', '.join(missing)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    else:
        print("\nAll required packages are installed!")
        return True


def main():
    print("Setting up CMSC723 Final Project...")
    print("=" * 50)
    
    print("\n1. Creating directory structure...")
    create_directories()
    
    print("\n2. Checking dependencies...")
    deps_ok = check_dependencies()
    
    print("\n" + "=" * 50)
    if deps_ok:
        print("Setup complete! You can now start collecting data.")
        print("\nNext steps:")
        print("1. Run: python src/data_collection/tweet_scraper.py")
        print("2. Run: python src/preprocessing/clean_tweets.py")
    else:
        print("Setup complete, but please install missing dependencies.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

