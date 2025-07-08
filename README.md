# Tetrix AI Agent with Feedback Loop

An AI agent that connects to real financial documents and automatically fixes errors using actual PDF data. This isn't just another rule-based system - it's an intelligent agent that reads the original documents, understands the financial context, and makes meaningful corrections.

## What This Actually Does

- Feeds with a document path and it goes to work
- Connects to Grant's analytics service to get the actual PDF data
- Finds mathematical discrepancies and suspicious patterns
- Uses AI to understand what the numbers should be based on the real document content  
- Calculates missing values from actual transaction history
- Validates fund-level totals against investment details
- Provides meaningful corrections based on corresponding document information

Think of it as having a really smart analyst who can read through financial documents and spot errors, except it processes everything in seconds instead of hours.

## The Breakthrough

The system now accesses the real parsed document content from Tetrix APIs. When it finds missing data like "total_invested: None", it doesn't just guess - it looks at the actual investment transactions, unrealized values, and other real data from the PDF to calculate what the number should be.

For example:
- Link Mobility was missing total_invested → calculated $113.3M from real PDF data
- Inspira was missing total_invested → calculated $307.5M from actual document values
- Fund NAV validation using real totals: $3.42B vs $3.42B (validated as correct)

## Quick Start

Need API key:

```bash
export OPENAI_API_KEY="api-key"
cd ~/ai-agent-v3
pip install -r requirements.txt
python main.py
```

The system automatically detects if you're connected to Grant's network and uses real data. If not, it falls back to mock mode for testing.

## How It Works

The magic happens in three steps:

First, it calls Grant's analytics service with a document path and gets back the full parsed PDF content - not just flags, but actual financial data including asset details, transaction history, fund totals, and investment metrics.

Then the AI agent analyzes discrepancies using this real data. For missing values, it performs actual calculations. For validation issues, it compares real numbers. For inconsistencies, it applies financial logic using the document context.

Finally, it provides meaningful corrections and re-validates to prove the improvements actually worked.

## Environment

```bash
OPENAI_API_KEY="api-key"
TETRIX_ANALYTICS_URL="http://internal-backen-micro-f3m5zasfnrzz-435617696.us-east-2.elb.amazonaws.com"
USE_MOCK_ANALYTICS="true"
```


## Core components handle different aspects

`analytics_client.py` manages the connection to Grant's service and extracts the real parsed document content from the APIs.

`discrepancy_processor.py` contains the AI agents that understand financial logic and perform calculations using real document data.

`financial_agent.py` provides the LLM interface for complex financial reasoning and validation.

`main.py` orchestrates the complete workflow and handles batch processing of multiple documents.