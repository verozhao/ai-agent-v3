# Tetrix AI Feedback Loop System

An AI-powered feedback loop system that integrates with Grant's analytics microservice to automatically detect and correct discrepancies in financial documents before consolidation.

## Overview

This system implements a workflow that takes extracted financial documents, identifies mathematical discrepancies and suspicious focus points through Grant's analytics API, and uses specialized AI agents to process and correct issues. The goal is to improve document accuracy before human review and consolidation.

The workflow follows this pattern:
Extracted Document → Grant's Analytics API → AI Processing → Document Improvement → Re-validation

## System Architecture

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

## Core Components

**analytics_client.py** - Interfaces with Grant's tetrix-analytics-microservice. Supports both real and mock analytics clients for testing without VPN connection.

**discrepancy_processor.py** - Specialized AI agents for different issue types including mathematical inconsistencies, suspicious data patterns, and combined processing workflows.

**feedback_loop_system.py** - Main orchestrator that manages the complete workflow from analytics integration through performance tracking and batch processing.

**evaluation_system.py** - Handles accuracy measurement and evaluation including JSON diffing, performance metrics, and detailed reporting.

**main.py** - Entry point that demonstrates integration with real financial documents and provides comprehensive system testing.

## Quick Start

### Prerequisites

Python 3.8+ is required. You'll need at least one LLM API key:

```bash
export OPENAI_API_KEY="your-openai-key"
# OR
export ANTHROPIC_API_KEY="your-anthropic-key"
```

For real analytics service access, ensure VPN connection to Grant's internal network. Optionally set:

```bash
export TETRIX_ANALYTICS_URL="http://internal-backen-micro-f3m5zasfnrzz-435617696.us-east-2.elb.amazonaws.com"
```

### Installation

```bash
cd ~/ai-agent-v3
pip install -r requirements.txt
```

### Usage

Run the main integration demo:
```bash
python main.py
```

This runs a sample integration test with real documents from Grant's analytics microservice. The system automatically detects whether to use real or mock analytics based on network connectivity.

For testing without VPN access:
```bash
export USE_MOCK_ANALYTICS="true"
python main.py
```

## Configuration

Environment variables control system behavior:

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

## Understanding the Results

The system processes documents in three phases:

**Analytics Phase** - Calls Grant's API with document path, receives discrepancies (mathematical errors with 95%+ confidence) and focus points (suspicious data with 70-90% confidence).

**AI Processing Phase** - Uses specialized processors to handle different issue types. Discrepancies get automatic corrections while focus points get careful analysis to determine if correction is needed.

**Re-validation Phase** - Calls Grant's API again with the improved document to measure actual reduction in issues and prove effectiveness.

Example output:
```
Issues Found: 76 (21 discrepancies, 55 focus points)
Corrections Applied: 21
Re-validation: 76 → 55 issues (21 resolved)
Actual Improvement: 27.6%
Processing Time: 1.25s
```

## Testing with Real Data

The system includes test cases based on real-world examples from Grant's analytics service including ABRY Partners fund documents. The integration validates end-to-end workflow with actual financial data and provides comprehensive error handling.

Test document paths follow the format: `PEFundPortfolioExtraction/{document_id}`

## Integration with Grant's Analytics

The system connects to Grant's tetrix-analytics-microservice which has already processed financial documents and identified potential issues. Both real and mock analytics clients are supported to enable development and testing without VPN access.

The analytics service returns structured data about mathematical inconsistencies and suspicious patterns that the AI agents then process for automatic correction or human review flagging.

## Troubleshooting

**Analytics service connection issues** - Ensure VPN connection to Grant's internal network, verify TETRIX_ANALYTICS_URL, or enable mock mode for testing.

**LLM processing errors** - Verify OPENAI_API_KEY or ANTHROPIC_API_KEY is set correctly. The system falls back to rule-based processing if LLM is unavailable.

**Low improvement scores** - Review document quality, check AI prompt effectiveness, validate that corrections are being applied properly.

**Debug information** - Check log output for detailed processing information including API calls, correction decisions, and performance metrics.

## Support

For analytics integration questions contact Grant. For AI/ML components and system architecture questions contact the development team. Production deployment should be coordinated with the infrastructure team for proper VPN and service configuration.