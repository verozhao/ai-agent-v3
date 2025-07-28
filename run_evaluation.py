"""
Run the comprehensive evaluation
"""

import subprocess
import sys

def main():
    print("Starting Comprehensive Evaluation...")
    print("This will evaluate 20 documents (5 training, 15 test)")
    print("="*60)
    
    # Check for OpenAI API key
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    # Run the evaluation
    subprocess.run([sys.executable, "evaluation_system.py"])

if __name__ == "__main__":
    main()