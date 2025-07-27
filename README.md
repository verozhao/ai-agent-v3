# Tetrix AI Agent v3 - Intelligent Financial Document Processing System

An advanced AI-powered system for processing and correcting financial documents using real LLM reasoning, multi-step validation, and autonomous decision-making capabilities.

## 🎯 What This System Does

This is a production-ready AI agent that:

- **Connects to real financial documents** via Tetrix Analytics APIs
- **Identifies discrepancies and data quality issues** using mathematical validation
- **Applies intelligent corrections** using OpenAI GPT-3.5 with financial domain expertise  
- **Validates improvements** by measuring actual document quality improvement
- **Learns from feedback** to improve future performance
- **Processes documents autonomously** with minimal human intervention

### Key Capabilities

- **Multi-step AI reasoning** with reflection and self-improvement
- **Real-time document processing** from Tetrix document extraction pipeline
- **Financial domain expertise** with knowledge of PE/VC fund structures
- **Asset-level corrections** for investment portfolios and fund data
- **Ground truth validation** against consolidated documents
- **Automated improvement measurement** showing percentage improvements

## 🏗️ System Architecture

### Core Components

```
main.py                    - Entry point and orchestration
├── feedback_loop.py       - Intelligent feedback loop system
├── document_agent.py      - Autonomous AI agent for corrections
├── ai_reasoning_engine.py - Multi-step LLM reasoning engine
├── analytics_client.py    - Tetrix API integration
├── financial_agent.py     - Financial domain specialist
└── pydantic_models.py     - Data validation and schemas
```

### Agent Architecture

The system implements --

1. **Document Agent** (`document_agent.py`) - Main autonomous agent
2. **Reasoning Engine** (`ai_reasoning_engine.py`) - Multi-step LLM reasoning
3. **Financial Specialist** (`financial_agent.py`) - Domain-specific knowledge
4. **Feedback Loop** (`feedback_loop.py`) - Continuous improvement system

## 🚀 Quick Start

### Prerequisites

```bash
# Required API Keys
export OPENAI_API_KEY="your-openai-key"
export TETRIX_ANALYTICS_URL="http://internal-backend-url"
export INTERNAL_API_KEY="your-internal-api-key"  # Optional

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run with real Tetrix API (production mode)
python main.py --mode production

# Run with mock data (testing mode) 
python main.py --mode testing --mock

# Process specific documents
python main.py --mode production
```

## 📋 Core Functions by File

### `main.py` - System Orchestration
- **`TetrixProductionSystem`** - Main system orchestrator
- **`initialize()`** - Set up all system components
- **`process_document()`** - Process single document through complete pipeline
- **`batch_process_documents()`** - Process multiple documents with analytics
- **`test_connectivity()`** - Verify API connections
- **`run_sample_integration_test()`** - End-to-end system test

### `document_agent.py` - Autonomous AI Agent
- **`DocumentAgent`** - Main AI agent class with autonomous decision-making
- **`process_document_intelligently()`** - Complete intelligent document processing
- **`_analyze_document_issues()`** - Identify all document issues
- **`_create_correction_plan()`** - AI-powered correction strategy planning
- **`_execute_intelligent_corrections()`** - Apply corrections with reasoning
- **`_validate_improvements()`** - Measure actual improvement results
- **`_learn_from_correction_process()`** - Continuous learning and adaptation

### `ai_reasoning_engine.py` - Multi-Step LLM Reasoning
- **`FinancialEngine`** - Advanced AI reasoning for financial documents
- **`reason_about_discrepancy()`** - 6-step reasoning process with reflection
- **`_analyze_discrepancy()`** - Step 1: Deep issue analysis
- **`_plan_correction()`** - Step 2: Intelligent correction planning
- **`_execute_correction()`** - Step 3: Apply correction with reasoning
- **`_verify_correction()`** - Step 4: Business logic validation
- **`_reflect_on_correction()`** - Step 5: Critical self-assessment (NEW)
- **`_improve_correction()`** - Step 6: Apply reflection-based improvements (NEW)

### `feedback_loop.py` - Intelligent Feedback System
- **`FeedbackLoopSystem`** - Complete AI-powered feedback loop
- **`process_document_with_feedback_loop()`** - End-to-end processing with validation
- **`_get_original_issues()`** - Extract document issues from analytics
- **`_measure_actual_improvement()`** - Calculate real improvement metrics
- **`_get_consolidated_document()`** - Access ground truth for validation
- **`batch_process_documents()`** - Process multiple documents with metrics

### `analytics_client.py` - Tetrix API Integration
- **`TetrixAnalyticsClient`** - Production API client for document analysis
- **`get_discrepancies_for_document()`** - Retrieve document issues from API
- **`get_raw_document_data()`** - Access full parsed document content
- **`revalidate_improved_document()`** - Validate corrections against API
- **`get_consolidated_documents()`** - Access ground truth data from MongoDB
- **`test_connection()`** - Verify API connectivity and VPN status

### `financial_agent.py` - Financial Domain Specialist
- **`FinancialDocumentSpecialistAgent`** - Expert financial document processor
- **`process_financial_document()`** - Comprehensive financial document analysis
- **`FinancialLLMEngine`** - LLM with financial domain expertise
- **`FinancialToolkit`** - Specialized financial calculations and validation
- **`validate_fund_name()`** - Fund name validation against dynamic registry
- **`calculate_irr()`** - Internal rate of return calculations
- **`validate_accounting_equation()`** - Fundamental accounting validation

### Supporting Components

#### `fund_registry_dynamic.py` - Dynamic Fund Registry
- **`DynamicFundRegistry`** - Real-time fund database from API
- **`fetch_fund_data_from_api()`** - Extract funds from document analysis
- **`find_fund_by_name()`** - Fuzzy matching for fund name validation
- **`refresh_fund_registry()`** - Update database with latest API data

#### `pydantic_models.py` - Data Validation
- **`ParsedDocumentModel`** - Generic document validation with financial rules
- **`GenericAssetModel`** - Asset-level data validation
- **`create_document_model_from_parsed_document()`** - Dynamic model creation
- **`validate_corrected_document()`** - Validation for corrected documents

#### `pdf_reader.py` - Direct PDF Analysis
- **`PDFDocumentReader`** - Direct PDF content extraction for validation
- **`read_pdf_for_validation()`** - Extract tables and text for cross-validation

## 🧠 AI Agent Intelligence

### Multi-Step Reasoning Process

The AI agent uses a sophisticated 6-step reasoning process:

1. **ANALYZE** - Deep analysis of what's wrong and why
2. **PLAN** - Create intelligent correction strategy  
3. **EXECUTE** - Apply correction with financial reasoning
4. **VERIFY** - Validate correction using business logic
5. **REFLECT** - Critical self-assessment of the correction (NEW)
6. **IMPROVE** - Apply improvements from reflection if needed (NEW)

### Reflection Pattern (New Enhancement)

The system now includes a reflection pattern where the AI:
- Critically analyzes its own corrections
- Identifies potential issues with mathematical accuracy
- Checks business logic and internal consistency
- Suggests improvements when confidence is low
- Re-applies corrections with enhanced reasoning

### Financial Domain Expertise

The agent has deep knowledge of:
- Private equity and venture capital fund structures
- Investment lifecycle analysis (fundraising → investment → exit)
- Financial statement analysis and accounting principles
- Fund reporting metrics (IRR, MOIC, DPI, RVPI)
- Asset-level investment tracking and valuations

## 📊 Performance & Results

### Current Performance Metrics

- **Documents Processed**: Handles PE/VC fund portfolio extraction documents
- **Issue Detection**: Identifies mathematical discrepancies and data quality issues
- **Correction Success**: Applies intelligent corrections with confidence scoring
- **Validation**: Measures actual improvement against ground truth data
- **Learning**: Adapts correction strategies based on feedback

### Example Results

```
BATCH PROCESSING RESULTS:
   Documents Processed: 1/1
   Total Issues Found: 6
   Corrections Applied: 4
   Improvement: 66.7%
   Processing Time: 15.2s
```

### Improvement Opportunities

Based on analysis, the system can be enhanced by:
1. **Ground Truth Integration** - Use consolidated documents for asset-specific corrections
2. **Enhanced Asset Context** - Provide more detailed asset information to the AI
3. **Calculation Validation** - Add real-time calculation verification
4. **Pattern Learning** - Store successful correction patterns for reuse

## 🔧 Configuration & Modes

### Operating Modes

- **Production Mode** (`--mode production`): Full AI reasoning, no ground truth
- **Testing Mode** (`--mode testing`): Enhanced AI reasoning with ground truth for evaluation  
- **Training Mode** (`--mode training`): Enhanced AI reasoning with ground truth for learning

### Environment Variables

```bash
# Core API Configuration
OPENAI_API_KEY=your-openai-key                    # Required for AI reasoning
ANTHROPIC_API_KEY=your-anthropic-key              # Alternative LLM (optional)
TETRIX_ANALYTICS_URL=http://analytics-service-url # Tetrix API endpoint
INTERNAL_API_KEY=your-internal-key                # Internal API authentication

# Database Configuration (for consolidated documents)
MONGODB_HOST=your-mongodb-host
MONGODB_USERNAME=your-mongodb-user  
MONGODB_PASSWORD=your-mongodb-password
MONGODB_DATABASE=your-database-name
MONGODB_COLLECTION=your-collection-name

# Optional Configuration
USE_MOCK_ANALYTICS=false                          # Use mock data instead of API
```

## 🔍 Debugging & Analysis

### Logging

The system provides comprehensive logging:

```bash
# View processing logs
tail -f tetrix_feedback_loop.log

# Key log patterns to watch for:
# - "Agent starting intelligent processing"
# - "Applied correction to [field] with [confidence]"
# - "Measured improvement: [original] → [remaining] issues"
```

### Performance Analysis

Use the included analysis tools:

```bash
# Analyze agent performance patterns
python analyze_agent_performance.py

# Suggest improvements with ground truth
python improve_agent_with_ground_truth.py
```

## 🧪 Testing

### Unit Tests

```bash
# Run basic tests
pytest

# Test specific components
python -m unittest test_improvement_fix.py
```

### Integration Tests

```bash
# Test full system integration
python main.py --mode testing

# Test with mock data
python main.py --mode testing --mock
```

### API Connectivity

```bash
# Test Tetrix API connection
python analytics_client.py

# Test fund registry 
python fund_registry_dynamic.py
```

## 📈 Future Enhancements

### Planned Improvements

1. **Enhanced Ground Truth Integration**
   - Direct asset-specific lookups from consolidated documents
   - Real-time validation against known correct values
   - Expected improvement: 2/121 → 50-80/121 corrections

2. **Advanced Calculation Engine**
   - Real-time IRR and MOIC calculations
   - Asset-level financial metric derivations
   - Cross-validation of mathematical relationships

3. **Pattern Learning System**
   - Store successful correction patterns
   - Learn from human feedback
   - Adapt to document-specific patterns

4. **Multi-Document Context**
   - Cross-document validation
   - Fund-level consistency checking
   - Historical data trend analysis

## 🤝 Contributing

### Development Setup

1. Clone repository and install dependencies
2. Set up environment variables for API access
3. Run tests to verify setup
4. Check connectivity to Tetrix APIs

### Code Organization

- **Core Agent Logic**: `document_agent.py`, `ai_reasoning_engine.py`
- **API Integration**: `analytics_client.py`, `fund_registry_dynamic.py`
- **Domain Knowledge**: `financial_agent.py`, `pydantic_models.py`
- **System Orchestration**: `main.py`, `feedback_loop.py`

### Adding New Features

1. **New Correction Types**: Add to `ai_reasoning_engine.py`
2. **Financial Rules**: Extend `financial_agent.py`
3. **Data Sources**: Integrate via `analytics_client.py`
4. **Validation Logic**: Add to `pydantic_models.py`

## 📄 License & Usage

This system is designed for processing financial documents in a production environment. It requires:

- Access to Tetrix Analytics APIs
- OpenAI API key for LLM reasoning
- VPN connectivity for internal services
- MongoDB access for ground truth validation

## 🔗 Related Documentation

- [Tetrix Analytics API Documentation](internal-docs-link)
- [Financial Document Processing Guide](internal-guide-link)
- [Agent Architecture Deep Dive](internal-architecture-link)

---

**Built with AI-first principles for autonomous financial document processing.**

*Last Updated: January 2025*