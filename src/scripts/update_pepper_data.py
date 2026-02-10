
import os
import sys
import subprocess
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_pipeline():
    """
    Orchestrate the update process:
    1. Augment/Ingest latest data (simulating fetching from extensive sources).
    2. Run standard data pipeline (if needed for other sources).
    3. Retrain model.
    """
    python_exe = sys.executable
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    logger.info("Starting Pepper Data Update Pipeline...")
    
    # 1. Fetch Real Data (scrapes source and appends to existing)
    logger.info("Step 1: Fetching Real Historical Data...")
    # First ensure we have base data
    ingest_script = os.path.join(script_dir, 'ingest_pepper.py')
    fetch_script = os.path.join(script_dir, 'fetch_real_pepper.py')
    
    try:
        # Reset base
        subprocess.run([python_exe, ingest_script], check=True)
        # Fetch new
        subprocess.run([python_exe, fetch_script], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Data update failed: {e}")
        return
    
    # 2. Retrain Model
    logger.info("Step 2: Retraining Pepper Model...")
    train_script = os.path.join(script_dir, 'train_pepper.py')
    try:
        subprocess.run([python_exe, train_script], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed: {e}")
        return
        
    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    update_pipeline()
