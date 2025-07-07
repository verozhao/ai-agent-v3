"""
Discrepancy and Focus Point Processor
Specialized AI agents for handling different types of financial data issues
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

from analytics_client import Discrepancy, FocusPoint, AnalyticsResponse
from financial_agent import FinancialCorrection, FinancialLLMEngine

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Result of processing a discrepancy or focus point"""
    issue_id: str
    issue_type: str
    field: str
    original_value: Any
    suggested_value: Any
    action_taken: str  # 'corrected', 'flagged', 'ignored', 'escalated'
    confidence: float
    reasoning: str
    evidence: List[str]
    processing_time: float
    llm_analysis: Optional[Dict[str, Any]] = None

class BaseIssueProcessor(ABC):
    """Base class for processing financial data issues"""
    
    def __init__(self, llm_engine: FinancialLLMEngine):
        self.llm_engine = llm_engine
        self.processing_history = []
        
    @abstractmethod
    async def process_issue(self, issue: Any, document: Dict[str, Any], 
                           historical_data: Dict[str, Any] = None) -> ProcessingResult:
        """Process a specific issue and return result"""
        pass
    
    def _create_financial_context(self, document: Dict[str, Any], 
                                historical_data: Dict[str, Any] = None) -> str:
        """Create financial context for LLM analysis"""
        context = f"""
FINANCIAL DOCUMENT CONTEXT:
Document Data: {json.dumps(document, indent=2)}

HISTORICAL DATA:
{json.dumps(historical_data or {}, indent=2)}

ANALYSIS FRAMEWORK:
- Apply financial domain expertise
- Use accounting principles and investment logic
- Consider mathematical consistency
- Evaluate business reasonableness
- Assess chronological logic
"""
        return context

class DiscrepancyProcessor(BaseIssueProcessor):
    """
    Processor for mathematical discrepancies (high confidence corrections)
    These are guaranteed to be wrong and need fixing
    """
    
    async def process_issue(self, discrepancy: Discrepancy, document: Dict[str, Any], 
                           historical_data: Dict[str, Any] = None) -> ProcessingResult:
        """Process a mathematical discrepancy with high confidence correction"""
        
        start_time = time.time()
        
        # For discrepancies, we have high confidence they're wrong
        # Focus on fixing rather than questioning
        
        correction_prompt = f"""
MATHEMATICAL DISCREPANCY CORRECTION

ISSUE DETAILS:
- Field: {discrepancy.field}
- Current Value: {discrepancy.current_value}
- Expected Value: {discrepancy.expected_value}
- Issue Type: {discrepancy.issue_type}
- Message: {discrepancy.message}
- Financial Rule: {discrepancy.financial_rule}
- Evidence: {discrepancy.evidence}

{self._create_financial_context(document, historical_data)}

This is a MATHEMATICAL DISCREPANCY - it violates fundamental financial principles.

CORRECTION TASK:
1. **VERIFY THE ISSUE**: Confirm this violates financial logic
2. **DETERMINE CORRECTION**: What should the value be?
3. **FINANCIAL REASONING**: Explain the financial principle being violated
4. **CORRECTION CONFIDENCE**: Rate your confidence (should be high for math errors)

RESPONSE FORMAT:
- Corrected Value: [specific value]
- Correction Type: [calculation, field_swap, format_fix, etc.]
- Financial Reasoning: [detailed explanation]
- Confidence: [0.0-1.0]
- Action: [corrected/flagged/escalated]

Focus on mathematical accuracy and financial consistency.
"""
        
        try:
            # Get LLM analysis
            llm_response = await self.llm_engine.reason_about_document(
                {"discrepancy_correction": correction_prompt},
                "discrepancy_correction"
            )
            
            processing_time = time.time() - start_time
            
            # Fetch actual document data and perform real calculations
            try:
                # First, get the source document data using REAL Tetrix API endpoints
                from analytics_client import create_analytics_client
                # Use real analytics client to call actual Tetrix API endpoints
                analytics_client = create_analytics_client(use_mock=False)
                
                logger.info(f"Fetching source document data for {discrepancy.field} correction...")
                
                # Get the document data 
                document_path = getattr(discrepancy, 'document_path', 'PEFundPortfolioExtraction/67ee89d7ecbb614e1103e533')
                
                async with analytics_client:
                    # Get the full document data AND the raw document
                    source_response = await analytics_client.get_discrepancies_for_document(document_path)
                    
                    # Try to get the original document data for calculations
                    try:
                        # Attempt to fetch the raw document data
                        raw_document_data = await analytics_client.get_raw_document_data(document_path)
                    except:
                        raw_document_data = "Raw document data not available"
                        
                # Create detailed analysis context with ALL available data
                other_discrepancies = [d for d in source_response.discrepancies if d.discrepancy_id != discrepancy.discrepancy_id]
                
                # Create intelligent correction prompt with REAL Tetrix document data
                import json
                
                # Extract relevant data from real Tetrix API response
                logger.info(f"Raw document data type: {type(raw_document_data)}")
                if isinstance(raw_document_data, dict):
                    logger.info(f"Raw document data keys: {list(raw_document_data.keys())}")
                    logger.info(f"Raw document data content (first 500 chars): {str(raw_document_data)[:500]}")
                
                # Extract real financial data from the parsed document
                has_financial_data = False
                assets_data = {}
                financial_totals = {}
                target_asset = None
                
                if isinstance(raw_document_data, dict):
                    # Look for assets list (this is the real data structure)
                    assets_list = raw_document_data.get('assets', [])
                    
                    # Convert assets list to dict for easier lookup
                    if assets_list:
                        assets_data = {asset.get('name', f'asset_{i}'): asset for i, asset in enumerate(assets_list)}
                    
                    # Extract financial totals from real document
                    financial_totals = {
                        'fund_name': raw_document_data.get('fund_name'),
                        'total_fund_net_asset_value': raw_document_data.get('total_fund_net_asset_value'),
                        'total_investments_unrealized_and_realized': raw_document_data.get('total_investments_unrealized_and_realized'),
                        'total_invested_capital': raw_document_data.get('total_invested_capital'),
                        'total_distribution_to_partners': raw_document_data.get('total_distribution_to_partners'),
                        'number_of_unrealized_investments': raw_document_data.get('number_of_unrealized_investments'),
                        'number_of_realized_investments': raw_document_data.get('number_of_realized_investments')
                    }
                    
                    # Find the specific asset if this is an asset-level discrepancy
                    if 'assets.' in discrepancy.field:
                        asset_name_parts = discrepancy.field.split('.')
                        if len(asset_name_parts) >= 2:
                            asset_name = asset_name_parts[1]
                            target_asset = assets_data.get(asset_name)
                            if target_asset:
                                logger.info(f"Found target asset '{asset_name}' for field '{discrepancy.field}'")
                    
                    # Check if we have meaningful financial data
                    has_financial_data = (
                        len(assets_data) > 0 or
                        any(v is not None for v in financial_totals.values()) or
                        'fund_name' in raw_document_data
                    )
                
                # If real API doesn't have financial data, fall back to using discrepancy context
                if not has_financial_data:
                    logger.warning(f"Real API response lacks financial data structure. Using discrepancy context for {discrepancy.field}")
                    
                    # Use the discrepancy's expected value and context
                    correction_prompt = f"""FINANCIAL CORRECTION TASK - Using discrepancy analysis:

DISCREPANCY DETAILS:
Field: {discrepancy.field}
Current Value: {discrepancy.current_value}
Expected Value: {discrepancy.expected_value}
Issue: {discrepancy.message}
Financial Rule Violated: {discrepancy.financial_rule}
Evidence: {discrepancy.evidence}
Confidence: {discrepancy.confidence}

CORRECTION LOGIC:
- Field: {discrepancy.field}
- Problem: {discrepancy.issue_type}
- Context: This is a {discrepancy.severity} severity discrepancy

TASK: 
Based on the financial rule violation and evidence, provide the corrected value for {discrepancy.field}.

For missing values (None/null): Provide "MISSING_DATA"
For calculation errors: Use the expected value if available
For consistency issues: Use the most logical/recent value

RESPOND WITH JUST THE CORRECTED VALUE:"""
                else:
                    # Create intelligent correction using real financial data
                    if target_asset and 'total_invested' in discrepancy.field:
                        # For missing total_invested, calculate from available asset data
                        asset_name = discrepancy.field.split('.')[1]
                        unrealized = target_asset.get('unrealized_value', 0)
                        realized = target_asset.get('realized_value', 0)
                        total_value = target_asset.get('total_value', 0)
                        moic = target_asset.get('gross_moic')
                        
                        correction_prompt = f"""FINANCIAL CALCULATION - Missing Total Invested for {asset_name}:

AVAILABLE REAL DATA FROM PDF:
- Unrealized Value: ${unrealized:,.0f}
- Realized Value: ${(realized or 0):,.0f}  
- Total Value: ${total_value:,.0f}
- Gross MOIC: {moic if moic is not None else 'N/A'}

CALCULATION METHODS:
1. If MOIC available: Total Invested = Total Value / MOIC
2. If no MOIC: Total Invested ≈ Total Value (for unrealized assets)
3. For realized assets: Total Invested = Original Cost Basis

Calculate the missing total_invested amount. Provide only the number."""

                    elif 'total_fund_net_asset_value' in discrepancy.field:
                        # For NAV discrepancy, compare with total investments
                        nav = financial_totals.get('total_fund_net_asset_value', 0)
                        investments = financial_totals.get('total_investments_unrealized_and_realized', 0)
                        
                        correction_prompt = f"""NAV VALIDATION - Fund Level Discrepancy:

REAL FUND DATA FROM PDF:
- Current NAV: ${nav:,.0f}
- Total Investments Value: ${investments:,.0f}
- Difference: ${abs(nav - investments):,.0f}

FINANCIAL RULE: NAV should approximately equal Total Investments Value
Issue: {discrepancy.message}

The current NAV (${nav:,.0f}) vs Investments (${investments:,.0f}) - validate if this is correct.
Provide the corrected NAV value or confirm current value."""

                    else:
                        # Generic correction using full document context
                        correction_prompt = f"""FINANCIAL CORRECTION - Using REAL document data:

ISSUE: {discrepancy.field} = {discrepancy.current_value}
PROBLEM: {discrepancy.message}

REAL FUND DATA:
{json.dumps(financial_totals, indent=2, default=str)}

TARGET ASSET DATA:
{json.dumps(target_asset, indent=2, default=str) if target_asset else 'Not an asset-level field'}

TASK: Based on the real financial data above, provide the corrected value for {discrepancy.field}.
Respond with just the corrected value."""
                
                # Direct OpenAI call with context
                import openai
                import os
                client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                
                response = await client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": correction_prompt}],
                    max_tokens=50,
                    temperature=0.1
                )
                
                corrected_value_text = response.choices[0].message.content.strip()
                
                # Clean up the response - extract just the value
                if "=" in corrected_value_text:
                    corrected_value_text = corrected_value_text.split("=")[-1].strip()
                if corrected_value_text.startswith('"') and corrected_value_text.endswith('"'):
                    corrected_value_text = corrected_value_text[1:-1]
                
                # Parse the response
                if corrected_value_text.upper() in ["NULL", "NONE"]:
                    suggested_value = None
                    action_taken = "corrected"
                    confidence = 0.85
                elif corrected_value_text.upper() in ["CANNOT_CORRECT", "UNKNOWN"]:
                    suggested_value = discrepancy.current_value
                    action_taken = "flagged"
                    confidence = 0.60
                elif corrected_value_text.upper() == "MISSING_DATA":
                    # Handle missing data by using expected value if available
                    suggested_value = discrepancy.expected_value if discrepancy.expected_value is not None else "MISSING_DATA"
                    action_taken = "corrected"
                    confidence = 0.75
                else:
                    # Try to parse as number first (remove $ and commas)
                    clean_text = corrected_value_text.replace("$", "").replace(",", "")
                    try:
                        suggested_value = float(clean_text)
                        action_taken = "corrected"
                        confidence = 0.90
                    except ValueError:
                        # It's a text value - if it's still "Corrected Value: X" format, extract the X
                        if ":" in corrected_value_text:
                            suggested_value = corrected_value_text.split(":")[-1].strip()
                        else:
                            suggested_value = corrected_value_text
                        action_taken = "corrected"
                        confidence = 0.85
                    
            except Exception as parse_error:
                logger.warning(f"Financial agent correction failed for {discrepancy.field}: {parse_error}")
                suggested_value = discrepancy.expected_value or discrepancy.current_value
                action_taken = "flagged"
                confidence = 0.50
            
            reasoning = f"LLM analysis: {llm_response['reasoning'][:200]}..."
            
            result = ProcessingResult(
                issue_id=discrepancy.discrepancy_id,
                issue_type="discrepancy",
                field=discrepancy.field,
                original_value=discrepancy.current_value,
                suggested_value=suggested_value,
                action_taken=action_taken,
                confidence=confidence,
                reasoning=reasoning,
                evidence=discrepancy.evidence,
                processing_time=processing_time,
                llm_analysis=llm_response
            )
            
            self.processing_history.append(result)
            
            logger.info(f"Processed discrepancy {discrepancy.discrepancy_id}: {action_taken} "
                       f"({confidence:.1%} confidence)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing discrepancy {discrepancy.discrepancy_id}: {e}")
            
            # Return escalation result on error
            return ProcessingResult(
                issue_id=discrepancy.discrepancy_id,
                issue_type="discrepancy",
                field=discrepancy.field,
                original_value=discrepancy.current_value,
                suggested_value=discrepancy.current_value,
                action_taken="escalated",
                confidence=0.0,
                reasoning=f"Processing error: {str(e)}",
                evidence=discrepancy.evidence,
                processing_time=time.time() - start_time
            )

class FocusPointProcessor(BaseIssueProcessor):
    """
    Processor for focus points (suspicious data that needs review)
    These may or may not be wrong - requires careful analysis
    """
    
    async def process_issue(self, focus_point: FocusPoint, document: Dict[str, Any], 
                           historical_data: Dict[str, Any] = None) -> ProcessingResult:
        """Process a focus point with careful analysis"""
        
        start_time = time.time()
        
        # For focus points, we need to determine if it's actually wrong
        # Could be unusual but correct data
        
        analysis_prompt = f"""
FOCUS POINT ANALYSIS

SUSPICIOUS DATA DETECTED:
- Field: {focus_point.field}
- Current Value: {focus_point.current_value}
- Flag Reason: {focus_point.flag_reason}
- Message: {focus_point.message}
- Historical Values: {focus_point.historical_values}
- Comparison Context: {focus_point.comparison_context}

{self._create_financial_context(document, historical_data)}

This is a FOCUS POINT - data that appears suspicious but may be correct.

ANALYSIS TASK:
1. **BUSINESS PLAUSIBILITY**: Could this value be legitimate business-wise?
2. **HISTORICAL CONTEXT**: How does it compare to historical patterns?
3. **FINANCIAL LOGIC**: Does it make sense given the financial context?
4. **CORRECTION NECESSITY**: Does this actually need correction?

POSSIBLE SCENARIOS:
- Legitimate business event (new funding round, major acquisition, etc.)
- Correct but unusual data (market conditions, one-time events)
- Data entry error requiring correction
- Field misalignment or format issue

RESPONSE FORMAT:
- Analysis: [Is this likely correct or incorrect?]
- Suggested Action: [correct/flag/ignore/request_review]
- Corrected Value: [if correction needed]
- Confidence: [0.0-1.0]
- Business Reasoning: [explain business context]

Be conservative - only suggest corrections if confident there's an error.
"""
        
        try:
            # Get LLM analysis
            llm_response = await self.llm_engine.reason_about_document(
                {"focus_point_analysis": analysis_prompt},
                "focus_point_analysis"
            )
            
            processing_time = time.time() - start_time
            
            # Use direct OpenAI call for focus point analysis
            try:
                # Create analysis prompt for this focus point
                analysis_prompt = f"""Analyze this financial data:

Field: {focus_point.field}
Current: {focus_point.current_value}
Issue: {focus_point.flag_reason}
Historical: {focus_point.historical_values}

Options:
- CORRECT: [value] - if you know right value
- FLAG - needs human review
- ACCEPT - current value OK

Answer:"""
                
                # Direct OpenAI call
                import openai
                import os
                client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                
                response = await client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": analysis_prompt}],
                    max_tokens=30,
                    temperature=0.1
                )
                
                response_text = response.choices[0].message.content.strip().upper()
                
                # Parse the agent's decision
                if response_text.startswith("CORRECT:"):
                    correction_value = response_text.replace("CORRECT:", "").strip()
                    try:
                        suggested_value = float(correction_value)
                    except ValueError:
                        suggested_value = correction_value
                    action_taken = "corrected"
                    confidence = 0.80
                elif response_text == "FLAG":
                    suggested_value = focus_point.current_value
                    action_taken = "flagged"
                    confidence = 0.70
                elif response_text == "ACCEPT":
                    suggested_value = focus_point.current_value
                    action_taken = "flagged"  # Still flag for review even if accepted
                    confidence = 0.60
                else:
                    # Default conservative
                    suggested_value = focus_point.current_value
                    action_taken = "flagged"
                    confidence = 0.50
                    
            except Exception as parse_error:
                logger.warning(f"Financial agent analysis failed for {focus_point.field}: {parse_error}")
                suggested_value = focus_point.current_value
                action_taken = "flagged"
                confidence = 0.50
            
            reasoning = f"LLM analysis: {llm_response['reasoning'][:200]}..."
            
            result = ProcessingResult(
                issue_id=focus_point.focus_point_id,
                issue_type="focus_point",
                field=focus_point.field,
                original_value=focus_point.current_value,
                suggested_value=suggested_value,
                action_taken=action_taken,
                confidence=confidence,
                reasoning=reasoning,
                evidence=[focus_point.flag_reason],
                processing_time=processing_time,
                llm_analysis=llm_response
            )
            
            self.processing_history.append(result)
            
            logger.info(f"Processed focus point {focus_point.focus_point_id}: {action_taken} "
                       f"({confidence:.1%} confidence)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing focus point {focus_point.focus_point_id}: {e}")
            
            return ProcessingResult(
                issue_id=focus_point.focus_point_id,
                issue_type="focus_point", 
                field=focus_point.field,
                original_value=focus_point.current_value,
                suggested_value=focus_point.current_value,
                action_taken="escalated",
                confidence=0.0,
                reasoning=f"Processing error: {str(e)}",
                evidence=[focus_point.flag_reason],
                processing_time=time.time() - start_time
            )
    
    def _should_auto_correct(self, focus_point: FocusPoint) -> bool:
        """Determine if focus point should be auto-corrected"""
        
        # Check for obvious patterns that indicate errors
        current_val = focus_point.current_value
        historical_vals = focus_point.historical_values
        
        if not historical_vals or len(historical_vals) < 2:
            return False
        
        # Check for decimal place errors (e.g., 165,000 vs 165,000,000)
        if isinstance(current_val, (int, float)) and all(isinstance(h, (int, float)) for h in historical_vals):
            avg_historical = sum(historical_vals) / len(historical_vals)
            
            # If current value is exactly 1000x or 0.001x historical average
            if abs(current_val - avg_historical * 1000) < avg_historical * 0.1:
                return True
            if abs(current_val - avg_historical / 1000) < avg_historical * 0.1:
                return True
        
        return False
    
    def _calculate_suggested_value(self, focus_point: FocusPoint) -> Any:
        """Calculate suggested correction value"""
        
        current_val = focus_point.current_value
        historical_vals = focus_point.historical_values
        
        if not historical_vals:
            return current_val
        
        # For decimal place errors, use historical average as baseline
        if isinstance(current_val, (int, float)) and all(isinstance(h, (int, float)) for h in historical_vals):
            avg_historical = sum(historical_vals) / len(historical_vals)
            
            # If current value seems 1000x too big
            if current_val > avg_historical * 100:
                return current_val / 1000
                
            # If current value seems 1000x too small
            if current_val < avg_historical / 100:
                return current_val * 1000
        
        return current_val

class CombinedIssueProcessor:
    """
    Combined processor that handles both discrepancies and focus points
    Orchestrates the specialized processors
    """
    
    def __init__(self, llm_engine: FinancialLLMEngine):
        self.llm_engine = llm_engine
        self.discrepancy_processor = DiscrepancyProcessor(llm_engine)
        self.focus_point_processor = FocusPointProcessor(llm_engine)
        self.processing_stats = {
            "total_issues": 0,
            "discrepancies_processed": 0,
            "focus_points_processed": 0,
            "corrections_applied": 0,
            "issues_flagged": 0,
            "issues_escalated": 0
        }
    
    async def process_analytics_response(self, analytics_response: AnalyticsResponse, 
                                       document: Dict[str, Any]) -> Dict[str, Any]:
        """Process all issues from analytics response"""
        
        start_time = time.time()
        results = {
            "document_path": analytics_response.document_path,
            "processing_timestamp": datetime.now().isoformat(),
            "discrepancy_results": [],
            "focus_point_results": [],
            "summary": {},
            "corrected_document": document.copy()
        }
        
        # Process discrepancies (high priority - mathematical errors)
        logger.info(f"Processing {len(analytics_response.discrepancies)} discrepancies...")
        
        for discrepancy in analytics_response.discrepancies:
            result = await self.discrepancy_processor.process_issue(
                discrepancy, document, analytics_response.historical_data
            )
            results["discrepancy_results"].append(asdict(result))
            
            # Apply high-confidence corrections
            if result.action_taken == "corrected" and result.confidence >= 0.85:
                results["corrected_document"][result.field] = result.suggested_value
                self.processing_stats["corrections_applied"] += 1
                logger.info(f"Applied correction: {result.field} = {result.suggested_value}")
            
            self.processing_stats["discrepancies_processed"] += 1
        
        # Process focus points (medium priority - suspicious data)
        logger.info(f"Processing {len(analytics_response.focus_points)} focus points...")
        
        for focus_point in analytics_response.focus_points:
            result = await self.focus_point_processor.process_issue(
                focus_point, document, analytics_response.historical_data
            )
            results["focus_point_results"].append(asdict(result))
            
            # Apply medium-confidence corrections for focus points
            if result.action_taken == "corrected" and result.confidence >= 0.80:
                results["corrected_document"][result.field] = result.suggested_value
                self.processing_stats["corrections_applied"] += 1
                logger.info(f"Applied focus point correction: {result.field} = {result.suggested_value}")
            elif result.action_taken == "flagged":
                self.processing_stats["issues_flagged"] += 1
            elif result.action_taken == "escalated":
                self.processing_stats["issues_escalated"] += 1
            
            self.processing_stats["focus_points_processed"] += 1
        
        self.processing_stats["total_issues"] += len(analytics_response.discrepancies) + len(analytics_response.focus_points)
        
        # Create summary
        total_processing_time = time.time() - start_time
        results["summary"] = {
            "total_issues": len(analytics_response.discrepancies) + len(analytics_response.focus_points),
            "discrepancies": len(analytics_response.discrepancies),
            "focus_points": len(analytics_response.focus_points),
            "corrections_applied": sum(1 for r in results["discrepancy_results"] + results["focus_point_results"] 
                                     if r["action_taken"] == "corrected"),
            "issues_flagged": sum(1 for r in results["discrepancy_results"] + results["focus_point_results"] 
                                if r["action_taken"] == "flagged"),
            "processing_time": total_processing_time,
            "document_improved": len([r for r in results["discrepancy_results"] + results["focus_point_results"] 
                                    if r["action_taken"] == "corrected"]) > 0
        }
        
        logger.info(f"Issue processing complete: {results['summary']['corrections_applied']} corrections, "
                   f"{results['summary']['issues_flagged']} flagged, {total_processing_time:.2f}s")
        
        return results
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self.processing_stats.copy()
        
        if stats["total_issues"] > 0:
            stats["correction_rate"] = stats["corrections_applied"] / stats["total_issues"]
            stats["flag_rate"] = stats["issues_flagged"] / stats["total_issues"]
            stats["escalation_rate"] = stats["issues_escalated"] / stats["total_issues"]
        else:
            stats["correction_rate"] = 0.0
            stats["flag_rate"] = 0.0
            stats["escalation_rate"] = 0.0
        
        return stats
    
    async def process_all_issues(self, discrepancies: List[Discrepancy], 
                                focus_points: List[FocusPoint], 
                                document: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process all discrepancies and focus points directly"""
        
        if document is None:
            document = {}
        
        start_time = time.time()
        results = {
            "processing_timestamp": datetime.now().isoformat(),
            "discrepancy_results": [],
            "focus_point_results": [],
            "corrections": [],
            "flagged_issues": [],
            "processing_successful": True
        }
        
        # Process discrepancies
        logger.info(f"Processing {len(discrepancies)} discrepancies...")
        
        for discrepancy in discrepancies:
            try:
                result = await self.discrepancy_processor.process_issue(
                    discrepancy, document, {}
                )
                results["discrepancy_results"].append(asdict(result))
                
                # Track corrections
                if result.action_taken == "corrected" and result.confidence >= 0.85:
                    results["corrections"].append({
                        "field": result.field,
                        "original_value": result.original_value,
                        "corrected_value": result.suggested_value,
                        "confidence": result.confidence,
                        "reasoning": result.reasoning
                    })
                    self.processing_stats["corrections_applied"] += 1
                
                self.processing_stats["discrepancies_processed"] += 1
                
            except Exception as e:
                logger.error(f"Error processing discrepancy {discrepancy.discrepancy_id}: {e}")
                results["processing_successful"] = False
        
        # Process focus points
        logger.info(f"Processing {len(focus_points)} focus points...")
        
        for focus_point in focus_points:
            try:
                result = await self.focus_point_processor.process_issue(
                    focus_point, document, {}
                )
                results["focus_point_results"].append(asdict(result))
                
                # Track corrections and flags
                if result.action_taken == "corrected" and result.confidence >= 0.80:
                    results["corrections"].append({
                        "field": result.field,
                        "original_value": result.original_value,
                        "corrected_value": result.suggested_value,
                        "confidence": result.confidence,
                        "reasoning": result.reasoning
                    })
                    self.processing_stats["corrections_applied"] += 1
                elif result.action_taken == "flagged":
                    results["flagged_issues"].append({
                        "field": result.field,
                        "issue": result.reasoning,
                        "confidence": result.confidence
                    })
                    self.processing_stats["issues_flagged"] += 1
                elif result.action_taken == "escalated":
                    self.processing_stats["issues_escalated"] += 1
                
                self.processing_stats["focus_points_processed"] += 1
                
            except Exception as e:
                logger.error(f"Error processing focus point {focus_point.focus_point_id}: {e}")
                results["processing_successful"] = False
        
        self.processing_stats["total_issues"] += len(discrepancies) + len(focus_points)
        
        total_processing_time = time.time() - start_time
        logger.info(f"Direct issue processing complete: {len(results['corrections'])} corrections, "
                   f"{len(results['flagged_issues'])} flagged, {total_processing_time:.2f}s")
        
        return results
    
    def reset_stats(self):
        """Reset processing statistics"""
        self.processing_stats = {
            "total_issues": 0,
            "discrepancies_processed": 0,
            "focus_points_processed": 0,
            "corrections_applied": 0,
            "issues_flagged": 0,
            "issues_escalated": 0
        }

# Export
__all__ = ["CombinedIssueProcessor", "DiscrepancyProcessor", "FocusPointProcessor", "ProcessingResult"]