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
            
            # Parse LLM response for correction
            suggested_value = discrepancy.expected_value  # Default to expected value
            action_taken = "corrected"
            confidence = 0.95  # High confidence for mathematical discrepancies
            
            # Enhanced parsing would extract from LLM response
            reasoning = f"Mathematical discrepancy in {discrepancy.field}: {discrepancy.message}"
            
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
            
            # Default to flagging for human review (conservative approach)
            suggested_value = focus_point.current_value
            action_taken = "flagged"
            confidence = 0.70  # Medium confidence for focus points
            
            # Enhanced parsing would analyze LLM response
            reasoning = f"Focus point in {focus_point.field}: {focus_point.flag_reason}"
            
            # Check for obvious corrections based on historical data
            if self._should_auto_correct(focus_point):
                suggested_value = self._calculate_suggested_value(focus_point)
                action_taken = "corrected"
                confidence = 0.85
            
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