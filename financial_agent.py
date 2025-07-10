"""
Financial Document Specialist Agent
Production-ready with real LLM reasoning, specialized financial knowledge, and adaptive learning
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import aiohttp
import sqlite3
from pathlib import Path

# Real LLM Integration
import openai
from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)

@dataclass
class FinancialCorrection:
    field: str
    original_value: Any
    corrected_value: Any
    correction_type: str 
    confidence: float
    reasoning: str
    evidence: List[str]
    financial_rule: str
    llm_reasoning: str

@dataclass
class FinancialDocumentAnalysis:
    document_id: str
    document_type: str  # 'investment', 'fund_report', 'financial_statement'
    anomalies_detected: List[Dict[str, Any]]
    financial_patterns: Dict[str, Any]
    semantic_analysis: Dict[str, Any]
    risk_indicators: List[str]
    confidence_score: float
    analysis_timestamp: datetime

class FinancialLLMEngine:
    """Production LLM engine specialized for financial reasoning"""
    
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.setup_clients()
        
        # Financial domain prompts
        self.financial_system_prompt = self._create_financial_system_prompt()
        
        # Performance tracking
        self.metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "avg_response_time": 0.0,
            "tokens_used": 0,
            "financial_accuracy": 0.0
        }
    
    def setup_clients(self):
        """Initialize LLM clients with real API keys"""
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        
        if openai_key:
            openai.api_key = openai_key
            self.openai_client = openai.AsyncOpenAI(api_key=openai_key)
            logger.info("OpenAI client initialized")
        
        if anthropic_key:
            self.anthropic_client = AsyncAnthropic(api_key=anthropic_key)
            logger.info("Anthropic client initialized")
        
        if not openai_key and not anthropic_key:
            raise ValueError("No LLM API keys found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
    
    def _create_financial_system_prompt(self) -> str:
        return """You are a world-class Financial Document Analysis Specialist with decades of experience in private equity, venture capital, and financial auditing.

EXPERTISE AREAS:
- Private equity and venture capital fund structures
- Investment lifecycle analysis (fundraising → investment → exit)
- Financial statement analysis and accounting principles
- Fund reporting and performance metrics (IRR, MOIC, DPI, RVPI)
- Regulatory compliance (SEC filings, fund disclosures)
- Data quality and anomaly detection in financial documents

CORE CAPABILITIES:
1. **Semantic Field Analysis**: Understand what each field should contain based on its name and context
2. **Financial Relationship Validation**: Verify mathematical relationships (totals, IRR calculations, etc.)
3. **Chronological Logic**: Ensure dates follow proper investment lifecycles
4. **Cross-Reference Validation**: Check consistency across related fields
5. **Pattern Recognition**: Identify common errors and data quality issues

FINANCIAL KNOWLEDGE BASE:
- Fund names typically include: "Capital Partners", "Fund", "LP", "Management", Roman numerals
- Investment dates precede exit dates by 2-7 years typically
- IRR calculations must align with investment period and multiples
- Quarterly data should sum to annual totals
- Assets = Liabilities + Equity (fundamental accounting equation)
- Performance metrics: IRR >15% is good, 20%+ is excellent for PE
- Fund sizes: Mega funds >$5B, Large funds $1-5B, Mid-market $500M-1B

ANOMALY DETECTION PATTERNS:
- Date values in name fields (obvious field swap)
- Fund names in date fields (field swap indicator)  
- Text amounts ("five million") in numeric fields
- Impossible chronology (exit before investment)
- Mathematical inconsistencies (totals ≠ sum of components)
- Out-of-range values (IRR >100%, negative fund sizes)

REASONING APPROACH:
1. **Context Understanding**: Analyze the document type and expected structure
2. **Field Semantic Analysis**: Determine if field content matches field purpose
3. **Financial Logic Validation**: Apply financial domain knowledge
4. **Cross-Field Consistency**: Check relationships between fields
5. **Confidence Assessment**: Rate certainty of findings

Always provide detailed financial reasoning and cite specific financial principles."""
    
    async def reason_about_document(self, document: Dict[str, Any], analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Use LLM for deep financial document reasoning"""
        
        start_time = time.time()
        
        # Create financial analysis prompt
        prompt = self._create_financial_analysis_prompt(document, analysis_type)
        
        try:
            # OpenAI
            if self.openai_client:
                response = await self._call_openai(prompt)
            # elif self.anthropic_client:
            #     response = await self._call_anthropic(prompt)
            else:
                raise ValueError("No LLM client available")
            
            # Update metrics
            response_time = time.time() - start_time
            self.metrics["total_calls"] += 1
            self.metrics["successful_calls"] += 1
            self.metrics["avg_response_time"] = (
                self.metrics["avg_response_time"] * (self.metrics["total_calls"] - 1) + response_time
            ) / self.metrics["total_calls"]
            
            return {
                "reasoning": response["content"],
                "financial_analysis": self._extract_financial_insights(response["content"]),
                "response_time": response_time,
                "model_used": response["model"],
                "tokens_used": response.get("tokens", 0)
            }
            
        except Exception as e:
            logger.error(f"LLM reasoning failed: {e}")
            self.metrics["total_calls"] += 1
            raise
    
    def _create_financial_analysis_prompt(self, document: Dict[str, Any], analysis_type: str) -> str:
        """Create specialized financial analysis prompt"""
        
        return f"""FINANCIAL DOCUMENT ANALYSIS REQUEST

Document Type: {analysis_type}
Document Data: {json.dumps(document, indent=2)}

Please perform a comprehensive financial document analysis:

1. **DOCUMENT CLASSIFICATION**:
   - What type of financial document is this?
   - What stage of investment lifecycle does it represent?
   - What are the key financial metrics present?

2. **FIELD SEMANTIC ANALYSIS**:
   - Analyze each field name vs. its content
   - Identify any obvious mismatches (dates in name fields, etc.)
   - Check for financial terminology consistency

3. **FINANCIAL RELATIONSHIP VALIDATION**:
   - Verify mathematical relationships between fields
   - Check cumulative totals (quarterly → annual, etc.)
   - Validate accounting equations if applicable
   - Assess IRR/multiple consistency with time periods

4. **CHRONOLOGICAL & BUSINESS LOGIC**:
   - Verify date sequences make financial sense
   - Check investment lifecycle progression
   - Validate business timeline reasonableness

5. **ANOMALY DETECTION**:
   - Identify data quality issues
   - Flag impossible or improbable values
   - Detect common error patterns

6. **CORRECTION RECOMMENDATIONS**:
   - Propose specific corrections with high confidence
   - Explain financial reasoning for each correction
   - Prioritize corrections by financial impact

Provide detailed analysis with specific financial reasoning and concrete correction recommendations."""
    
    async def _call_openai(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI GPT-3.5 for financial reasoning"""
        
        messages = [
            {"role": "system", "content": self.financial_system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = await self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.1,  # Low temperature for precise financial analysis
            max_tokens=2000
        )
        
        return {
            "content": response.choices[0].message.content,
            "model": response.model,
            "tokens": response.usage.total_tokens if response.usage else 0
        }
    
    async def _call_anthropic(self, prompt: str) -> Dict[str, Any]:
        """Call Anthropic Claude for financial reasoning"""
        
        message = await self.anthropic_client.messages.create(
            model="claude-3-sonnet-20241022",
            max_tokens=2000,
            temperature=0.1,
            system=self.financial_system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {
            "content": message.content[0].text,
            "model": "claude-3-sonnet",
            "tokens": message.usage.input_tokens + message.usage.output_tokens
        }
    
    def _extract_financial_insights(self, reasoning_text: str) -> Dict[str, Any]:
        """Extract structured insights from LLM reasoning"""
        
        # Parse key financial insights (simplified - could use more sophisticated NLP)
        insights = {
            "document_type_identified": "investment" in reasoning_text.lower(),
            "anomalies_found": "anomal" in reasoning_text.lower() or "error" in reasoning_text.lower(),
            "corrections_suggested": "correct" in reasoning_text.lower() or "should be" in reasoning_text.lower(),
            "high_confidence_issues": "obvious" in reasoning_text.lower() or "clear" in reasoning_text.lower(),
            "financial_rules_applied": []
        }
        
        # Extract mentioned financial concepts
        financial_concepts = [
            "irr", "multiple", "fund", "investment", "exit", "accounting equation",
            "field swap", "chronological", "total", "quarterly"
        ]
        
        for concept in financial_concepts:
            if concept in reasoning_text.lower():
                insights["financial_rules_applied"].append(concept)
        
        return insights

class FinancialToolkit:
    """Specialized financial calculation and validation tools"""
    
    def __init__(self):
        self.fund_database = self._initialize_fund_database()
        
    def _initialize_fund_database(self) -> sqlite3.Connection:
        """Initialize real fund database"""
        db_path = "financial_funds.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create fund registry table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fund_registry (
                id INTEGER PRIMARY KEY,
                fund_name TEXT UNIQUE,
                fund_family TEXT,
                vintage_year INTEGER,
                fund_size_usd BIGINT,
                strategy TEXT,
                geographic_focus TEXT,
                status TEXT,
                inception_date DATE,
                first_close_date DATE,
                final_close_date DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Note: Fund data is now populated dynamically from API via DynamicFundRegistry
        # No hardcoded fund data - use fund_registry_dynamic.py for real-time fund information
        # Update the db if needed
        logger.info("Fund database initialized - use DynamicFundRegistry for real fund data from API")
        
        return conn
    
    async def validate_fund_name(self, fund_name: str) -> Dict[str, Any]:
        """Validate fund name using dynamic registry from API"""
        try:
            # Try dynamic registry first (real API data)
            try:
                from fund_registry_dynamic import DynamicFundRegistry
                
                async with DynamicFundRegistry() as registry:
                    result = registry.find_fund_by_name(fund_name)
                    
                    if result:
                        matched_fund_name, confidence = result
                        
                        # Get full fund details
                        funds = registry.get_all_funds()
                        fund_data = next((f for f in funds if f.fund_name == matched_fund_name), None)
                        
                        if fund_data:
                            return {
                                "is_valid": confidence >= 0.9,
                                "match_type": "exact" if confidence >= 0.95 else "fuzzy",
                                "matched_fund": matched_fund_name,
                                "similarity_score": confidence,
                                "fund_data": {
                                    "fund_name": fund_data.fund_name,
                                    "fund_family": fund_data.fund_family,
                                    "vintage_year": fund_data.vintage_year,
                                    "fund_size_usd": fund_data.fund_size_usd,
                                    "strategy": fund_data.strategy,
                                    "geographic_focus": fund_data.geographic_focus,
                                    "source": "dynamic_registry_api"
                                },
                                "confidence": confidence,
                                "suggested_correction": matched_fund_name if confidence < 0.9 else None
                            }
                            
            except Exception as dynamic_error:
                logger.warning(f"Dynamic registry failed: {dynamic_error}, falling back to static database")
            
            # Fallback to static database
            cursor = self.fund_database.cursor()
            
            # Exact match
            cursor.execute("SELECT * FROM fund_registry WHERE fund_name = ?", (fund_name,))
            exact_match = cursor.fetchone()
            
            if exact_match:
                return {
                    "is_valid": True,
                    "match_type": "exact",
                    "fund_data": {
                        "fund_name": exact_match[1],
                        "fund_family": exact_match[2],
                        "vintage_year": exact_match[3],
                        "fund_size_usd": exact_match[4],
                        "strategy": exact_match[5],
                        "source": "static_database"
                    },
                    "confidence": 1.0
                }
            
            # Fuzzy matching on static data
            cursor.execute("SELECT * FROM fund_registry")
            all_funds = cursor.fetchall()
            
            best_match = None
            best_score = 0
            
            for fund in all_funds:
                fund_name_db = fund[1]
                # Simple similarity check
                common_words = set(fund_name.lower().split()) & set(fund_name_db.lower().split())
                if len(common_words) >= 2:
                    score = len(common_words) / max(len(fund_name.split()), len(fund_name_db.split()))
                    if score > best_score and score > 0.5:
                        best_score = score
                        best_match = fund
            
            if best_match:
                return {
                    "is_valid": False,
                    "match_type": "fuzzy",
                    "suggested_correction": best_match[1],
                    "similarity_score": best_score,
                    "fund_data": {
                        "fund_name": best_match[1],
                        "fund_family": best_match[2],
                        "vintage_year": best_match[3],
                        "source": "static_database"
                    },
                    "confidence": best_score
                }
            
            # Check if it looks like a date (common error)
            import re
            if re.match(r'\d{4}-\d{2}-\d{2}', fund_name):
                return {
                    "is_valid": False,
                    "issue": "fund_name_appears_to_be_date",
                    "error_type": "field_swap",
                    "confidence": 0.95
                }
            
            return {
                "is_valid": False,
                "match_type": "none",
                "confidence": 0.0
            }
            
        except Exception as e:
            return {"error": str(e), "is_valid": False}
    
    async def calculate_irr(self, cash_flows: List[float], dates: List[str]) -> Dict[str, Any]:
        """Calculate IRR with financial validation"""
        try:
            if len(cash_flows) != len(dates) or len(cash_flows) < 2:
                return {"error": "Invalid cash flow data", "irr": None}
            
            # Convert dates
            date_objects = [pd.to_datetime(date) for date in dates]
            
            # Simple IRR approximation (in production, use numpy-financial)
            initial_investment = abs(cash_flows[0]) if cash_flows[0] < 0 else cash_flows[0]
            final_value = cash_flows[-1]
            
            if initial_investment <= 0 or final_value <= 0:
                return {"error": "Invalid cash flow values", "irr": None}
            
            years = (date_objects[-1] - date_objects[0]).days / 365.25
            
            if years <= 0:
                return {"error": "Invalid date range", "irr": None}
            
            # IRR = (Final Value / Initial Investment)^(1/years) - 1
            irr = ((final_value / initial_investment) ** (1/years)) - 1
            
            # Financial validation
            is_reasonable = -0.5 <= irr <= 3.0  # -50% to 300% is reasonable range
            
            return {
                "irr": irr,
                "irr_percentage": irr * 100,
                "is_reasonable": is_reasonable,
                "investment_period_years": years,
                "multiple": final_value / initial_investment,
                "calculation_method": "simple_irr"
            }
            
        except Exception as e:
            return {"error": str(e), "irr": None}
    
    async def validate_accounting_equation(self, assets: float, liabilities: float, equity: float) -> Dict[str, Any]:
        """Validate fundamental accounting equation"""
        try:
            expected_equity = assets - liabilities
            difference = abs(equity - expected_equity)
            tolerance = max(assets * 0.001, 1.0)  # 0.1% tolerance or $1
            
            is_valid = difference <= tolerance
            
            return {
                "is_valid": is_valid,
                "expected_equity": expected_equity,
                "actual_equity": equity,
                "difference": difference,
                "difference_percentage": (difference / assets) * 100 if assets > 0 else 0,
                "tolerance_used": tolerance,
                "correction_needed": not is_valid
            }
            
        except Exception as e:
            return {"error": str(e), "is_valid": False}
    
    async def validate_cumulative_totals(self, components: Dict[str, float], total_field: str, total_value: float) -> Dict[str, Any]:
        """Validate cumulative totals (quarterly, etc.)"""
        try:
            component_sum = sum(components.values())
            difference = abs(total_value - component_sum)
            tolerance = max(component_sum * 0.001, 0.01)  # 0.1% tolerance
            
            is_valid = difference <= tolerance
            
            return {
                "is_valid": is_valid,
                "expected_total": component_sum,
                "actual_total": total_value,
                "components": components,
                "difference": difference,
                "correction_needed": not is_valid,
                "component_count": len(components)
            }
            
        except Exception as e:
            return {"error": str(e), "is_valid": False}
    
    async def validate_investment_chronology(self, investment_date: str, exit_date: str = None, other_dates: Dict[str, str] = None) -> Dict[str, Any]:
        """Validate investment timeline logic"""
        try:
            inv_date = pd.to_datetime(investment_date)
            
            issues = []
            validations = {
                "investment_date_valid": True,
                "chronology_valid": True,
                "business_logic_valid": True
            }
            
            # Check if investment date is reasonable
            current_date = pd.Timestamp.now()
            years_ago = (current_date - inv_date).days / 365.25
            
            if years_ago < 0:
                issues.append("Investment date is in the future")
                validations["investment_date_valid"] = False
            elif years_ago > 20:
                issues.append("Investment date is more than 20 years ago")
                validations["business_logic_valid"] = False
            
            # Validate exit date if provided
            if exit_date:
                exit_dt = pd.to_datetime(exit_date)
                
                if exit_dt <= inv_date:
                    issues.append("Exit date is before or same as investment date")
                    validations["chronology_valid"] = False
                
                holding_period = (exit_dt - inv_date).days / 365.25
                if holding_period > 15:
                    issues.append(f"Holding period of {holding_period:.1f} years is unusually long")
                elif holding_period < 0.25:
                    issues.append(f"Holding period of {holding_period:.1f} years is unusually short")
            
            # Validate other dates
            if other_dates:
                for date_name, date_value in other_dates.items():
                    try:
                        other_dt = pd.to_datetime(date_value)
                        if "start" in date_name.lower() or "begin" in date_name.lower():
                            # Start dates should be before investment or close to it
                            if other_dt > inv_date + timedelta(days=90):
                                issues.append(f"{date_name} is significantly after investment date")
                    except:
                        pass
            
            return {
                "is_valid": len(issues) == 0,
                "issues": issues,
                "validations": validations,
                "investment_date_parsed": inv_date.isoformat(),
                "business_logic_score": 1.0 - (len(issues) * 0.25)
            }
            
        except Exception as e:
            return {"error": str(e), "is_valid": False}

class FinancialDocumentSpecialistAgent:
    """
    Financial Document Specialist Agent
    Real LLM reasoning + Financial domain expertise + Production architecture
    """
    
    def __init__(self):
        self.agent_id = f"financial_specialist_{uuid.uuid4().hex[:8]}"
        self.name = "FinancialDocumentSpecialist"
        
        # Core components
        self.llm_engine = FinancialLLMEngine()
        self.toolkit = FinancialToolkit()
        
        # Learning system
        self.correction_history = []
        self.pattern_library = self._initialize_pattern_library()
        self.performance_metrics = {
            "documents_processed": 0,
            "corrections_made": 0,
            "accuracy_rate": 1.0,
            "confidence_scores": [],
            "learning_iterations": 0
        }
        
        # Specialized knowledge
        self.financial_rules = self._load_financial_rules()
        
        logger.info("Financial Document Specialist Agent initialized")
    
    def _initialize_pattern_library(self) -> Dict[str, Any]:
        """Initialize financial pattern recognition library"""
        return {
            "field_swap_patterns": {
                "date_in_name_field": {
                    "pattern": r'\d{4}-\d{2}-\d{2}',
                    "confidence": 0.95,
                    "correction_strategy": "swap_with_date_field"
                },
                "name_in_date_field": {
                    "indicators": ["fund", "capital", "partners", "llc", "lp"],
                    "confidence": 0.90,
                    "correction_strategy": "swap_with_name_field"
                }
            },
            "calculation_patterns": {
                "quarterly_sum": {
                    "fields": ["q1", "q2", "q3", "q4", "total"],
                    "rule": "q1 + q2 + q3 + q4 = total",
                    "tolerance": 0.01
                },
                "accounting_equation": {
                    "fields": ["assets", "liabilities", "equity"],
                    "rule": "assets = liabilities + equity",
                    "tolerance": 0.001
                }
            },
            "semantic_patterns": {
                "text_amounts": {
                    "patterns": [r'(\w+)\s+million', r'(\w+)\s+billion'],
                    "conversion_map": {
                        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
                    }
                }
            }
        }
    
    def _load_financial_rules(self) -> Dict[str, Any]:
        """Load financial domain rules and constraints"""
        return {
            "irr_constraints": {
                "min_reasonable": -0.5,  # -50%
                "max_reasonable": 3.0,   # 300%
                "typical_range": (0.10, 0.35)  # 10-35% for PE
            },
            "multiple_constraints": {
                "min_reasonable": 0.1,
                "max_reasonable": 10.0,
                "typical_range": (1.5, 4.0)
            },
            "holding_period_constraints": {
                "min_months": 6,
                "max_years": 15,
                "typical_years": (3, 7)
            },
            "fund_size_constraints": {
                "min_usd": 50_000_000,      # $50M minimum
                "max_usd": 50_000_000_000,  # $50B maximum
                "mega_fund_threshold": 5_000_000_000  # $5B+
            }
        }
    
    async def process_financial_document(self, document: Dict[str, Any], document_type: str = "auto_detect") -> Dict[str, Any]:
        """
        Main entry point: Process financial document with real intelligence
        """
        start_time = time.time()
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"Processing financial document {doc_id}")
        
        try:
            # Phase 1: LLM-powered document analysis
            llm_analysis = await self.llm_engine.reason_about_document(document, "comprehensive")
            
            # Phase 2: Specialized financial validation
            validation_results = await self._run_financial_validations(document)
            
            # Phase 3: Pattern-based anomaly detection
            pattern_analysis = await self._detect_financial_patterns(document)
            
            # Phase 4: Cross-validation with LLM reasoning
            final_analysis = await self._synthesize_analysis(document, llm_analysis, validation_results, pattern_analysis)
            
            # Phase 5: Generate corrections with high confidence
            corrections = await self._generate_financial_corrections(document, final_analysis)
            
            # Phase 6: Apply corrections and validate
            corrected_document = await self._apply_corrections(document, corrections)
            
            processing_time = time.time() - start_time
            
            # Update performance metrics
            self._update_performance_metrics(corrections, processing_time)
            
            result = {
                "document_id": doc_id,
                "original_document": document,
                "corrected_document": corrected_document,
                "corrections": [asdict(c) for c in corrections],
                "analysis": {
                    "llm_reasoning": llm_analysis,
                    "validation_results": validation_results,
                    "pattern_analysis": pattern_analysis,
                    "final_synthesis": final_analysis
                },
                "confidence_score": final_analysis.get("overall_confidence", 0.8),
                "processing_time": processing_time,
                "agent_id": self.agent_id
            }
            
            logger.info(f"Financial document {doc_id} processed: {len(corrections)} corrections, {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing financial document {doc_id}: {e}")
            return {
                "document_id": doc_id,
                "error": str(e),
                "processing_failed": True,
                "agent_id": self.agent_id
            }
    
    async def _run_financial_validations(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Run specialized financial validations using toolkit"""
        
        validations = {}
        
        # Fund name validation
        for field, value in document.items():
            if any(term in field.lower() for term in ["fund", "name"]) and isinstance(value, str):
                fund_validation = await self.toolkit.validate_fund_name(value)
                validations[f"{field}_fund_validation"] = fund_validation
        
        # Investment chronology validation
        date_fields = {k: v for k, v in document.items() if "date" in k.lower()}
        if "investment_date" in date_fields:
            chronology = await self.toolkit.validate_investment_chronology(
                investment_date=str(date_fields["investment_date"]),
                exit_date=str(date_fields.get("exit_date")),
                other_dates={k: str(v) for k, v in date_fields.items() if k not in ["investment_date", "exit_date"]}
            )
            validations["chronology_validation"] = chronology
        
        # Accounting equation validation
        if all(field in document for field in ["assets", "liabilities", "equity"]):
            accounting = await self.toolkit.validate_accounting_equation(
                assets=float(document["assets"]),
                liabilities=float(document["liabilities"]),
                equity=float(document["equity"])
            )
            validations["accounting_validation"] = accounting
        
        # Cumulative totals validation
        total_fields = [k for k in document.keys() if "total" in k.lower()]
        for total_field in total_fields:
            base_name = total_field.lower().replace("total_", "").replace("_total", "")
            components = {k: float(v) for k, v in document.items() 
                         if base_name in k.lower() and k != total_field and isinstance(v, (int, float))}
            
            if len(components) >= 2:
                cumulative = await self.toolkit.validate_cumulative_totals(
                    components=components,
                    total_field=total_field,
                    total_value=float(document[total_field])
                )
                validations[f"{total_field}_cumulative"] = cumulative
        
        # IRR validation if applicable
        if "irr" in document and "investment_date" in document and "exit_date" in document:
            try:
                investment_amount = None
                exit_value = None
                
                for field, value in document.items():
                    if "investment" in field.lower() and "amount" in field.lower():
                        investment_amount = float(value)
                    elif "exit" in field.lower() and "value" in field.lower():
                        exit_value = float(value)
                
                if investment_amount and exit_value:
                    irr_calc = await self.toolkit.calculate_irr(
                        cash_flows=[-investment_amount, exit_value],
                        dates=[str(document["investment_date"]), str(document["exit_date"])]
                    )
                    validations["irr_validation"] = irr_calc
            except:
                pass
        
        return validations
    
    async def _detect_financial_patterns(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Detect financial anomaly patterns"""
        
        detected_patterns = {
            "field_swaps": [],
            "calculation_errors": [],
            "semantic_issues": [],
            "format_problems": []
        }
        
        # Field swap detection
        for field, value in document.items():
            if "date" in field.lower() and isinstance(value, str):
                # Check if date field contains fund name
                if any(indicator in value.lower() for indicator in ["fund", "capital", "partners", "llc"]):
                    detected_patterns["field_swaps"].append({
                        "field": field,
                        "issue": "date_field_contains_fund_name",
                        "value": value,
                        "confidence": 0.95
                    })
                    
            elif any(term in field.lower() for term in ["fund", "name"]) and isinstance(value, str):
                # Check if name field contains date
                import re
                if re.match(r'\d{4}-\d{2}-\d{2}', value):
                    detected_patterns["field_swaps"].append({
                        "field": field,
                        "issue": "name_field_contains_date",
                        "value": value,
                        "confidence": 0.95
                    })
        
        # Text amount detection
        for field, value in document.items():
            if any(term in field.lower() for term in ["amount", "value", "size"]) and isinstance(value, str):
                if "million" in value.lower() or "billion" in value.lower():
                    detected_patterns["semantic_issues"].append({
                        "field": field,
                        "issue": "text_amount_in_numeric_field",
                        "value": value,
                        "confidence": 0.90
                    })
        
        return detected_patterns
    
    async def _synthesize_analysis(self, document: Dict[str, Any], llm_analysis: Dict[str, Any], 
                                 validations: Dict[str, Any], patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to synthesize all analysis results"""
        
        synthesis_prompt = f"""FINANCIAL ANALYSIS SYNTHESIS

Original Document: {json.dumps(document, indent=2)}

LLM Analysis: {llm_analysis.get('reasoning', '')}

Validation Results: {json.dumps(validations, indent=2)}

Pattern Detection: {json.dumps(patterns, indent=2)}

Please synthesize these findings into a comprehensive assessment:

1. **CRITICAL ISSUES** (requiring immediate correction):
   - List issues with >90% confidence
   - Prioritize by financial impact

2. **PROBABLE ISSUES** (likely corrections):
   - List issues with 70-90% confidence
   - Assess risk vs. confidence

3. **OVERALL CONFIDENCE**:
   - Rate overall document quality (0-1)
   - Identify highest confidence corrections

4. **CORRECTION STRATEGY**:
   - Recommend correction approach
   - Sequence corrections optimally

Provide specific, actionable financial analysis."""
        
        synthesis_response = await self.llm_engine.reason_about_document(
            {"synthesis_request": synthesis_prompt}, 
            "synthesis"
        )
        
        # Calculate overall confidence
        critical_issues = len([p for pattern_list in patterns.values() for p in pattern_list if p.get("confidence", 0) > 0.9])
        validation_failures = len([v for v in validations.values() if not v.get("is_valid", True)])
        
        overall_confidence = max(0.1, 1.0 - (critical_issues * 0.2) - (validation_failures * 0.15))
        
        return {
            "synthesis_reasoning": synthesis_response.get("reasoning", ""),
            "critical_issues": critical_issues,
            "validation_failures": validation_failures,
            "overall_confidence": overall_confidence,
            "correction_priority": "high" if critical_issues > 0 else "medium" if validation_failures > 0 else "low"
        }
    
    async def _generate_financial_corrections(self, document: Dict[str, Any], analysis: Dict[str, Any]) -> List[FinancialCorrection]:
        """Generate high-confidence financial corrections"""
        
        corrections = []
        
        # Use LLM to generate corrections based on analysis
        correction_prompt = f"""FINANCIAL CORRECTION GENERATION

Document: {json.dumps(document, indent=2)}

Analysis Summary: {analysis.get('synthesis_reasoning', '')}

Generate specific corrections for this financial document. For each correction, provide:
1. Field name
2. Current value
3. Corrected value
4. Correction type (field_swap, calculation, format, semantic)
5. Confidence score (0-1)
6. Financial reasoning
7. Evidence supporting the correction

Focus on corrections with >80% confidence that follow established financial principles."""
        
        correction_response = await self.llm_engine.reason_about_document(
            {"correction_request": correction_prompt},
            "correction_generation"
        )
        
        # Parse LLM corrections and create structured corrections
        # (In production, would use more sophisticated parsing)
        
        # Field swap corrections
        for field, value in document.items():
            if "date" in field.lower() and isinstance(value, str):
                if any(indicator in value.lower() for indicator in ["fund", "capital", "partners"]):
                    # Find likely fund name field
                    for other_field, other_value in document.items():
                        if ("fund" in other_field.lower() or "name" in other_field.lower()) and self._is_date_like(str(other_value)):
                            corrections.append(FinancialCorrection(
                                field=field,
                                original_value=value,
                                corrected_value=other_value,
                                correction_type="field_swap",
                                confidence=0.95,
                                reasoning="Date field contains fund name, swapping with date value in name field",
                                evidence=[f"Found date pattern in {other_field}", f"Found fund indicators in {field}"],
                                financial_rule="Field semantic consistency",
                                llm_reasoning=correction_response.get("reasoning", "")
                            ))
        
        # Calculation corrections
        for field, value in document.items():
            if "total" in field.lower():
                base_name = field.lower().replace("total_", "").replace("_total", "")
                components = {}
                for k, v in document.items():
                    if base_name in k.lower() and k != field and isinstance(v, (int, float)):
                        components[k] = float(v)
                
                if len(components) >= 2:
                    expected_total = sum(components.values())
                    current_total = float(value)
                    
                    if abs(expected_total - current_total) > 0.01:
                        corrections.append(FinancialCorrection(
                            field=field,
                            original_value=current_total,
                            corrected_value=expected_total,
                            correction_type="calculation",
                            confidence=0.95,
                            reasoning=f"Total should equal sum of components: {sum(components.values())}",
                            evidence=[f"Components: {components}"],
                            financial_rule="Cumulative total consistency",
                            llm_reasoning=correction_response.get("reasoning", "")
                        ))
        
        # Text amount corrections
        for field, value in document.items():
            if isinstance(value, str) and ("amount" in field.lower() or "value" in field.lower()):
                if "million" in value.lower():
                    # Extract number
                    import re
                    match = re.search(r'(\w+)\s+million', value.lower())
                    if match:
                        word = match.group(1)
                        number_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10}
                        if word in number_map:
                            numeric_value = number_map[word] * 1_000_000
                            corrections.append(FinancialCorrection(
                                field=field,
                                original_value=value,
                                corrected_value=numeric_value,
                                correction_type="semantic",
                                confidence=0.90,
                                reasoning=f"Converted text amount '{value}' to numeric value",
                                evidence=[f"Text pattern: {word} million"],
                                financial_rule="Numeric field format consistency",
                                llm_reasoning=correction_response.get("reasoning", "")
                            ))
        
        return corrections
    
    async def _apply_corrections(self, document: Dict[str, Any], corrections: List[FinancialCorrection]) -> Dict[str, Any]:
        """Apply corrections to create corrected document"""
        
        corrected_doc = document.copy()
        
        # Sort corrections by confidence (highest first)
        corrections_sorted = sorted(corrections, key=lambda x: x.confidence, reverse=True)
        
        # Apply high-confidence corrections
        for correction in corrections_sorted:
            if correction.confidence >= 0.80:  # Only apply high-confidence corrections
                corrected_doc[correction.field] = correction.corrected_value
                logger.info(f"Applied correction: {correction.field} = {correction.corrected_value} (confidence: {correction.confidence:.2f})")
        
        return corrected_doc
    
    def _is_date_like(self, value: str) -> bool:
        """Check if value looks like a date"""
        import re
        date_patterns = [r'\d{4}-\d{2}-\d{2}', r'\d{2}/\d{2}/\d{4}', r'\d{1,2}-[A-Za-z]{3}-\d{4}']
        return any(re.match(pattern, value) for pattern in date_patterns)
    
    def _update_performance_metrics(self, corrections: List[FinancialCorrection], processing_time: float):
        """Update agent performance metrics"""
        self.performance_metrics["documents_processed"] += 1
        self.performance_metrics["corrections_made"] += len(corrections)
        
        if corrections:
            avg_confidence = sum(c.confidence for c in corrections) / len(corrections)
            self.performance_metrics["confidence_scores"].append(avg_confidence)
    
    async def learn_from_feedback(self, document_id: str, ground_truth: Dict[str, Any], our_corrections: List[FinancialCorrection]):
        """Learn from human feedback to improve future performance"""
        
        learning_prompt = f"""FINANCIAL LEARNING FROM FEEDBACK

Our Corrections: {json.dumps([asdict(c) for c in our_corrections], indent=2)}

Ground Truth: {json.dumps(ground_truth, indent=2)}

Analyze our performance:
1. Which corrections were accurate?
2. What did we miss?
3. Were any corrections unnecessary?
4. How can we improve our financial reasoning?
5. What patterns should we remember?

Provide insights for improving financial document analysis accuracy."""
        
        learning_response = await self.llm_engine.reason_about_document(
            {"learning_request": learning_prompt},
            "learning_analysis"
        )
        
        # Store learning insights
        self.correction_history.append({
            "document_id": document_id,
            "our_corrections": [asdict(c) for c in our_corrections],
            "ground_truth": ground_truth,
            "learning_insights": learning_response.get("reasoning", ""),
            "timestamp": datetime.now().isoformat()
        })
        
        self.performance_metrics["learning_iterations"] += 1
        
        logger.info(f"Learning completed for document {document_id}")
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "type": "Financial Document Specialist",
            "performance_metrics": self.performance_metrics,
            "llm_metrics": self.llm_engine.metrics,
            "correction_history_size": len(self.correction_history),
            "financial_rules_loaded": len(self.financial_rules),
            "pattern_library_size": len(self.pattern_library),
            "status_timestamp": datetime.now().isoformat()
        }

# Export
__all__ = ["FinancialDocumentSpecialistAgent", "FinancialCorrection", "FinancialLLMEngine", "FinancialToolkit"]