"""
Main Entrypoint for Adaptive Preprocessing Selection for Domain-Specific NLP Pipelines
Authors: Ritesh Sharma (ENG25CSE012), Longjam Nirish Singh (ENG25CSE031)
"""

import sys
import os
import time

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.policy_engine import DomainPolicyEngine, POLICY_NAMES, POLICY_DESCRIPTIONS
from src.profiler import LocalMetaProfiler
from src.partitioner import MicroClusterPartitioner
from src.router import LightGBMSurrogateRouter
from src.gateway import AdaptivePreprocessingGateway
from src.evaluation import run_all


def banner():
    print("""
================================================================================
  ADAPTIVE PREPROCESSING SELECTION FOR DOMAIN-SPECIFIC NLP PIPELINES
  A Meta-Learning Approach
  Team Members:
    - Ritesh Sharma         USN: ENG25CSE012
    - Longjam Nirish Singh  USN: ENG25CSE031
================================================================================
    """)


def main():
    banner()

    print("Step 1: Validating Directory Structure and Model Calibration...")
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Initialize gateway (calibrates router if not present)
    gateway = AdaptivePreprocessingGateway()

    print("\nStep 2: Executing Full Experimental Suite...")
    run_all()

    print("\n" + "=" * 80)
    print("🚀 All benchmark experiments passed successfully!")
    print("To launch the interactive Streamlit Web Demonstrator, run:")
    print("    python -m streamlit run app.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
