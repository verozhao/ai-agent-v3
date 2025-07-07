# Tetrix AI Feedback Loop System

An AI-powered feedback loop system that integrates with Grant's analytics microservice to automatically detect and correct discrepancies in financial documents before consolidation.

## 🎯 Overview

This system implements a workflow for:

1. **Document Extraction** → **Analytics API** → **AI Processing** → **Document Improvement**
2. Identifying mathematical discrepancies and suspicious focus points
3. Using specialized AI agents to process and correct issues
4. Providing comprehensive evaluation and accuracy measurement
5. Generating detailed reports for continuous improvement

## 🏗️ System Architecture

```
Extracted Document → Grant's Analytics API → AI Feedback Loop System
         │                        │
         └─────────────┬──────────┘
                       │
             Discrepancies & Focus Points
                       │
             AI Correction & Evaluation
                       │
                 Improved Document
```

## 📁 Core Components

- **analytics_client.py**: Interfaces with Grant's tetrix-analytics-microservice. Supports both real and mock analytics clients (mock mode for testing without VPN).
- **discrepancy_processor.py**: Specialized AI agents for different issue types (mathematical inconsistencies, suspicious data, combined processing).
- **feedback_loop_system.py**: Main orchestrator for the complete workflow, including integration with analytics API, performance tracking, and batch processing.
- **evaluation_system.py**: Advanced accuracy measurement and evaluation, including JSON diffing, performance metrics, and reporting.
- **test_suite.py**: Comprehensive testing with real-world scenarios, error handling, and workflow validation.

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **LLM API Keys** (set one):
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   # OR
   export ANTHROPIC_API_KEY="your-anthropic-key"
   ```
3. **VPN Connection** (for real analytics service)
4. **Analytics Service URL** (optional):
   ```bash
   export TETRIX_ANALYTICS_URL="http://internal-backen-micro-f3m5zasfnrzz-435617696.us-east-2.elb.amazonaws.com"
   ```

### Installation

```bash
cd ai-agent-v3
pip install -r requirements.txt
```

### Usage

#### 1. Integration Demo (main system)
```bash
python main.py
```

- This runs a sample integration test with real documents and Grant's analytics microservice.
- By default, the system tries to use the real analytics service. To force mock mode, set the environment variable:
  ```bash
  export USE_MOCK_ANALYTICS="true"
  ```

#### 2. Comprehensive Test Suite
```bash
python test_suite.py
```
- Use `--mock` to run tests with the mock analytics client (no VPN required):
  ```bash
  python test_suite.py --mock
  ```
- Use `--verbose` for detailed logging.

## 🔧 Configuration

Set environment variables as needed:

```bash
# LLM Configuration
OPENAI_API_KEY="your-openai-api-key"
ANTHROPIC_API_KEY="your-anthropic-api-key"

# Analytics Service
TETRIX_ANALYTICS_URL="http://internal-backen-micro-f3m5zasfnrzz-435617696.us-east-2.elb.amazonaws.com"
USE_MOCK_ANALYTICS="false"  # Set to "true" for testing without VPN

# System Configuration
ENVIRONMENT="development"  # development, staging, production
```

## 📊 Understanding the Results

- **Analytics Phase**: Calls Grant's API with document path, receives discrepancies and focus points.
- **AI Processing Phase**: Handles discrepancies (mathematical errors) and focus points (suspicious data).
- **Improvement Measurement**: Accuracy against ground truth, improvement score, processing time.

Example output:
```
📊 Issues Found: 3
🔧 Corrections Applied: 2
📈 Improvement Score: 85.0%
⏱️  Processing Time: 1.25s
```

## 🧪 Testing with Real Data

- Test cases are included based on real-world examples (e.g., Abry Partners Fund Report, Solamere Fund II Capital Call).
- The test suite validates end-to-end workflow and error handling.

## 🔗 Integration with Grant's Analytics

- The system uses Grant's analytics microservice for discrepancy and focus point detection.
- Both real and mock analytics clients are supported (mock mode for offline/local testing).

## 🆘 Troubleshooting

- **Analytics service not connected**: Ensure VPN is connected, check `TETRIX_ANALYTICS_URL`, or use mock mode.
- **LLM not available**: Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.
- **Low accuracy scores**: Review test data, check LLM prompt effectiveness, validate document parsing logic.
- **Debug mode**: Use `--verbose` flag for detailed logs.

## 📞 Support

- **Analytics Integration**: Contact Grant
- **AI/ML Components**: Contact Veronica
- **System Architecture**: Review this documentation
- **Production Deployment**: Coordinate with infrastructure team