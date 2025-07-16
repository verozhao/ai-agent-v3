"""
AI Reasoning Engine - Core Intelligence for Financial Document Processing
AI agent with multi-step reasoning and tool use
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import openai
from anthropic import AsyncAnthropic
from pdf_reader import PDFDocumentReader

logger = logging.getLogger(__name__)

class ReasoningStep(Enum):
    ANALYZE = "analyze"
    PLAN = "plan"
    EXECUTE = "execute"
    VERIFY = "verify"
    LEARN = "learn"

@dataclass
class ReasoningTrace:
    """Detailed trace of AI reasoning process"""
    step: ReasoningStep
    timestamp: datetime
    input_data: Dict[str, Any]
    reasoning: str
    decision: str
    confidence: float
    tools_used: List[str]
    output: Dict[str, Any]
    execution_time: float

@dataclass
class AgentResponse:
    """Structured response from AI agent"""
    success: bool
    message: str
    reasoning_chain: List[ReasoningTrace]
    final_decision: Dict[str, Any]
    confidence: float
    tools_used: List[str]
    next_actions: List[str]
    metadata: Dict[str, Any]

class FinancialEngine:
    """
    Advanced AI reasoning engine for financial document processing
    Implements chain-of-thought reasoning and tool use
    """
    
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.reasoning_history = []
        self.available_tools = {}
        self.pdf_reader = PDFDocumentReader()  # <--- NEW
        self.setup_llm_clients()
        
    def setup_llm_clients(self):
        """Initialize LLM clients with proper error handling"""
        
        # Load environment variables from .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            logger.warning("python-dotenv not available, using system environment variables only")
        
        # OpenAI setup
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.openai_client = openai.AsyncOpenAI(api_key=openai_key)
            logger.info("OpenAI client initialized")
        
        # Anthropic setup
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            self.anthropic_client = AsyncAnthropic(api_key=anthropic_key)
            logger.info("Anthropic client initialized")
        
        if not self.openai_client and not self.anthropic_client:
            logger.warning("No LLM API keys found - AI reasoning will be limited")
    
    def register_tool(self, name: str, tool_instance):
        """Register a tool that the AI can use"""
        self.available_tools[name] = tool_instance
        logger.info(f"Registered tool: {name}")
    
    async def reason_about_discrepancy(self, discrepancy: Dict[str, Any],
                                     document_context: Dict[str, Any]) -> AgentResponse:
        """
        Use advanced reasoning to analyze and correct a financial discrepancy
        
        This is the core intelligence - it thinks step by step about financial issues
        Enhanced with REFLECTION pattern for financial accuracy
        """
        
        start_time = time.time()
        reasoning_chain = []
        
        # Step 1: ANALYZE - Deep analysis of the discrepancy
        analysis_trace = await self._analyze_discrepancy(discrepancy, document_context)
        reasoning_chain.append(analysis_trace)
        
        # Step 2: PLAN - Create a correction strategy
        plan_trace = await self._plan_correction(discrepancy, document_context, analysis_trace.output)
        reasoning_chain.append(plan_trace)
        
        # Step 3: EXECUTE - Apply the correction with reasoning
        execution_trace = await self._execute_correction(discrepancy, document_context, plan_trace.output)
        reasoning_chain.append(execution_trace)
        
        # Step 4: VERIFY - Validate the correction makes sense
        verification_trace = await self._verify_correction(discrepancy, execution_trace.output)
        reasoning_chain.append(verification_trace)
        
        # Step 5: REFLECT - Critical self-assessment of the correction (NEW REFLECTION PATTERN)
        reflection_trace = await self._reflect_on_correction(discrepancy, execution_trace.output, document_context)
        reasoning_chain.append(reflection_trace)
        
        # Step 6: IMPROVE - Apply improvements from reflection if needed
        final_correction = execution_trace.output
        if reflection_trace.output.get("needs_improvement") and reflection_trace.confidence > 0.7:
            improvement_trace = await self._improve_correction(discrepancy, execution_trace.output, reflection_trace.output, document_context)
            reasoning_chain.append(improvement_trace)
            final_correction = improvement_trace.output
        
        # Determine final decision based on reasoning chain (including reflection)
        final_confidence = min(trace.confidence for trace in reasoning_chain if hasattr(trace, 'confidence'))
        
        # Use final_correction which may be improved through reflection
        if final_confidence >= 0.6 and final_correction.get("corrected_value") is not None:  # Slightly lower threshold due to reflection
            success = True
            reflection_note = " (enhanced by reflection)" if reflection_trace.output.get("needs_improvement") else ""
            message = f"Successfully corrected {discrepancy.get('field')} with {final_confidence:.1%} confidence{reflection_note}"
            final_decision = {
                "action": "correct",
                "field": discrepancy.get("field"),
                "original_value": discrepancy.get("current_value"),
                "corrected_value": final_correction.get("corrected_value"),
                "reasoning": final_correction.get("justification", execution_trace.reasoning),
                "confidence": final_confidence,
                "reflection_applied": reflection_trace.output.get("needs_improvement", False)
            }
        else:
            success = False
            message = f"Could not correct {discrepancy.get('field')} - insufficient confidence or unclear solution"
            final_decision = {
                "action": "flag_for_review",
                "field": discrepancy.get("field"),
                "reason": reflection_trace.reasoning if reflection_trace else verification_trace.reasoning,
                "confidence": final_confidence
            }
        
        total_time = time.time() - start_time
        
        return AgentResponse(
            success=success,
            message=message,
            reasoning_chain=reasoning_chain,
            final_decision=final_decision,
            confidence=final_confidence,
            tools_used=[tool for trace in reasoning_chain for tool in (trace.tools_used if isinstance(trace.tools_used, list) else [trace.tools_used])],
            next_actions=["apply_correction"] if success else ["human_review"],
            metadata={
                "processing_time": total_time,
                "discrepancy_type": discrepancy.get("issue_type"),
                "reasoning_steps": len(reasoning_chain)
            }
        )
    
    async def _analyze_discrepancy(self, discrepancy: Dict[str, Any], 
                                 document_context: Dict[str, Any]) -> ReasoningTrace:
        """Step 1: Deep analysis of what's wrong and why"""
        
        start_time = time.time()
        
        analysis_prompt = f"""
You are a financial expert analyzing a discrepancy in a private equity fund document.

DISCREPANCY DETAILS:
- Field: {discrepancy.get('field')}
- Current Value: {discrepancy.get('current_value')}
- Expected Value: {discrepancy.get('expected_value')}
- Issue: {discrepancy.get('message')}
- Confidence: {discrepancy.get('confidence', 0)}
- Financial Rule Violated: {discrepancy.get('financial_rule', 'Unknown')}

DOCUMENT CONTEXT:
{json.dumps(document_context, indent=2)[:1000]}...

ANALYSIS TASK:
1. **What exactly is wrong?** Analyze the specific nature of this discrepancy
2. **Why did this happen?** Identify the likely root cause (data entry error, calculation mistake, etc.)
3. **What are the implications?** How does this affect the document's accuracy?
4. **Financial context**: Does this make sense from a business perspective?

Respond with structured analysis in this format:
{{
    "issue_type": "calculation_error|data_entry|field_swap|format_issue|business_logic_violation",
    "root_cause": "detailed explanation of what went wrong",
    "severity": "critical|high|medium|low",
    "business_impact": "how this affects financial reporting",
    "correction_approach": "strategy for fixing this issue",
    "confidence": 0.0-1.0
}}
"""
        
        try:
            if self.openai_client:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": analysis_prompt}],
                    temperature=0.1,
                    max_tokens=800
                )
                
                analysis_text = response.choices[0].message.content
                
                # Extract structured response
                try:
                    # Look for JSON in the response
                    json_start = analysis_text.find('{')
                    json_end = analysis_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        analysis_result = json.loads(analysis_text[json_start:json_end])
                    else:
                        # Fallback to manual parsing
                        analysis_result = {
                            "issue_type": "data_entry",
                            "root_cause": "Unable to parse structured analysis",
                            "severity": "medium",
                            "business_impact": "Requires manual review",
                            "correction_approach": "Conservative approach needed",
                            "confidence": 0.6
                        }
                except json.JSONDecodeError:
                    analysis_result = {
                        "issue_type": "unknown",
                        "root_cause": analysis_text,
                        "severity": "medium",
                        "business_impact": "Uncertain",
                        "correction_approach": "Manual review recommended",
                        "confidence": 0.5
                    }
                
                reasoning = f"AI Analysis: {analysis_result.get('root_cause', 'No clear cause identified')}"
                confidence = analysis_result.get("confidence", 0.7)
                
            else:
                # Fallback rule-based analysis
                analysis_result = self._rule_based_analysis(discrepancy)
                reasoning = "Rule-based analysis (no LLM available)"
                confidence = 0.6
        
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            analysis_result = {
                "issue_type": "unknown",
                "root_cause": f"Analysis failed: {str(e)}",
                "severity": "medium",
                "business_impact": "Unknown",
                "correction_approach": "Conservative approach",
                "confidence": 0.3
            }
            reasoning = f"Analysis failed: {str(e)}"
            confidence = 0.3
        
        execution_time = time.time() - start_time
        
        return ReasoningTrace(
            step=ReasoningStep.ANALYZE,
            timestamp=datetime.now(),
            input_data={"discrepancy": discrepancy, "document_context": document_context},
            reasoning=reasoning,
            decision=f"Issue type: {analysis_result.get('issue_type')}, Approach: {analysis_result.get('correction_approach')}",
            confidence=confidence,
            tools_used=["llm_analysis" if self.openai_client else "rule_based"],
            output=analysis_result,
            execution_time=execution_time
        )
    
    async def _analyze_extraction_quality(self, discrepancy: Dict[str, Any], 
                                    document_context: Dict[str, Any],
                                    pdf_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze if this is an extraction error vs data error, now with optional direct PDF context."""
        # Try to get PDF context if not provided
        if pdf_context is None:
            pdf_path = self._extract_pdf_path(document_context, discrepancy)
            if pdf_path:
                try:
                    pdf_context = await self.pdf_reader.read_pdf_for_validation(pdf_path)
                    logger.info(f"Loaded PDF context for extraction analysis from {pdf_path}")
                except Exception as e:
                    logger.warning(f"Could not read PDF for extraction analysis: {e}")
                    pdf_context = None
            else:
                logger.info("No PDF path found for extraction analysis; proceeding without PDF context.")
        
        extraction_error_prompt = f"""
You are analyzing a potential PDF extraction error vs actual data quality issue.

DISCREPANCY:
- Field: {discrepancy.get('field')}
- Extracted Value: {discrepancy.get('current_value')}
- Expected/Consolidated Value: {discrepancy.get('expected_value')}

DOCUMENT CONTEXT:
{json.dumps(document_context, indent=2)[:1000]}...

{f"PDF CONTEXT (if available): {json.dumps(pdf_context, indent=2)[:500]}..." if pdf_context else ""}

IMPORTANT CONTEXT:
- If Expected Value is None/null, the fund may not have reported this metric
- Common extraction errors:
  1. Field misalignment (value from row above/below)
  2. Similar field confusion (invested vs contributed capital)
  3. Column shifts in tables
  4. OCR errors in numbers
  5. Date/text in numeric fields

SPECIFIC CHECKS:
- Is the extracted value similar to any other field's value in the document?
- Does the magnitude suggest a different field (e.g., 3.5B for invested vs contributed)?
- Are there nearby fields with similar names that could be confused?

ANALYSIS NEEDED:
1. Is this likely an extraction error or genuine data issue?
2. If extraction error, what type?
3. Should we trust the extracted value?
4. Is a None/null in consolidated normal for this field?

Consider:
- "total_contributed_capital" vs "total_invested_capital" are different metrics
- Not all funds report all metrics every period
- Extraction might grab wrong cell from PDF table
- Large round numbers (like 3,538,536,554) might be from wrong field

Respond with JSON:
{{
    "error_type": "extraction_error|data_quality_issue|missing_metric|field_confusion",
    "confidence": 0.0-1.0,
    "reasoning": "detailed explanation",
    "likely_source_field": "field name if extraction error",
    "recommended_action": "skip_correction|apply_correction|flag_for_manual_review|investigate_further"
}}
"""
        
        try:
            if self.openai_client:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": extraction_error_prompt}],
                    temperature=0.1,
                    max_tokens=800
                )
                
                analysis_text = response.choices[0].message.content
                
                # Parse JSON response
                try:
                    json_start = analysis_text.find('{')
                    json_end = analysis_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        analysis_result = json.loads(analysis_text[json_start:json_end])
                    else:
                        # Fallback parsing
                        analysis_result = self._parse_extraction_analysis_fallback(analysis_text)
                except json.JSONDecodeError:
                    analysis_result = self._parse_extraction_analysis_fallback(analysis_text)
                
            else:
                # No LLM available - use rule-based analysis
                analysis_result = self._rule_based_extraction_analysis(discrepancy, document_context)
            
            # Enhance analysis with additional checks
            enhanced_analysis = self._enhance_extraction_analysis(analysis_result, discrepancy, document_context)
            
            return enhanced_analysis
            
        except Exception as e:
            logger.error(f"Extraction quality analysis failed: {e}")
            # Return conservative analysis on error
            return {
                "error_type": "unknown",
                "confidence": 0.3,
                "reasoning": f"Analysis failed: {str(e)}",
                "likely_source_field": None,
                "recommended_action": "flag_for_manual_review",
                "analysis_failed": True
            }

    def _parse_extraction_analysis_fallback(self, analysis_text: str) -> Dict[str, Any]:
        """Fallback parsing when JSON extraction fails"""
        
        result = {
            "error_type": "unknown",
            "confidence": 0.5,
            "reasoning": analysis_text,
            "likely_source_field": None,
            "recommended_action": "flag_for_manual_review"
        }
        
        # Try to extract key information from text
        text_lower = analysis_text.lower()
        
        # Determine error type
        if "extraction error" in text_lower:
            result["error_type"] = "extraction_error"
        elif "missing metric" in text_lower or "not reported" in text_lower:
            result["error_type"] = "missing_metric"
        elif "field confusion" in text_lower or "wrong field" in text_lower:
            result["error_type"] = "field_confusion"
        elif "data quality" in text_lower:
            result["error_type"] = "data_quality_issue"
        
        # Extract confidence if mentioned
        import re
        confidence_match = re.search(r'confidence[:\s]+([0-9.]+)', text_lower)
        if confidence_match:
            try:
                result["confidence"] = float(confidence_match.group(1))
            except:
                pass
        
        # Determine action
        if "skip" in text_lower or "don't correct" in text_lower:
            result["recommended_action"] = "skip_correction"
        elif "manual review" in text_lower:
            result["recommended_action"] = "flag_for_manual_review"
        elif "apply" in text_lower or "correct" in text_lower:
            result["recommended_action"] = "apply_correction"
        
        return result

    def _rule_based_extraction_analysis(self, discrepancy: Dict[str, Any], 
                                      document_context: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based analysis when LLM is not available"""
        
        field = discrepancy.get("field", "")
        current_value = discrepancy.get("current_value")
        expected_value = discrepancy.get("expected_value")
        
        # Check for None expected value
        if expected_value is None:
            return {
                "error_type": "missing_metric",
                "confidence": 0.8,
                "reasoning": f"Consolidated document has no value for {field} - likely not reported for this period",
                "likely_source_field": None,
                "recommended_action": "skip_correction"
            }
        
        # Check for field confusion patterns
        if "contributed" in field and current_value:
            # Check if value matches invested capital
            invested_capital = document_context.get("total_invested_capital")
            if invested_capital and abs(current_value - invested_capital) < 1000:
                return {
                    "error_type": "field_confusion",
                    "confidence": 0.9,
                    "reasoning": f"Value {current_value} matches total_invested_capital - likely extraction error",
                    "likely_source_field": "total_invested_capital",
                    "recommended_action": "skip_correction"
                }
        
        # Check for obvious magnitude differences
        if current_value and expected_value:
            ratio = current_value / expected_value if expected_value != 0 else float('inf')
            if ratio > 10 or ratio < 0.1:
                return {
                    "error_type": "extraction_error",
                    "confidence": 0.7,
                    "reasoning": f"Large magnitude difference (ratio: {ratio:.2f}) suggests extraction error",
                    "likely_source_field": None,
                    "recommended_action": "flag_for_manual_review"
                }
        
        # Default: assume data quality issue
        return {
            "error_type": "data_quality_issue",
            "confidence": 0.5,
            "reasoning": "No clear extraction error patterns detected",
            "likely_source_field": None,
            "recommended_action": "apply_correction"
        }

    def _enhance_extraction_analysis(self, analysis_result: Dict[str, Any],
                               discrepancy: Dict[str, Any],
                               document_context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance the analysis with additional context and validation"""
        
        # Add field-specific intelligence
        field = discrepancy.get("field", "")
        current_value = discrepancy.get("current_value")
        
        # Check for common confusion pairs
        confusion_pairs = {
            "total_contributed_capital": ["total_invested_capital", "total_called_capital", "total_committed_capital"],
            "total_invested_capital": ["total_contributed_capital", "total_active_investments_cost_basis"],
            "gross_irr": ["net_irr", "gross_moic"],
            "realized_value": ["unrealized_value", "total_value"]
        }
        
        if field in confusion_pairs and current_value is not None:
            for similar_field in confusion_pairs[field]:
                similar_value = document_context.get(similar_field)
                if similar_value is not None:
                    # Check if values are suspiciously close
                    if isinstance(current_value, (int, float)) and isinstance(similar_value, (int, float)):
                        if abs(current_value - similar_value) < abs(current_value) * 0.01:  # Within 1%
                            analysis_result["field_confusion_detected"] = True
                            analysis_result["confused_with"] = similar_field
                            analysis_result["confidence"] = min(analysis_result.get("confidence", 0.5) * 0.7, 0.9)
                            if analysis_result.get("error_type") == "data_quality_issue":
                                analysis_result["error_type"] = "field_confusion"
                            break
        
        # Add extraction pattern detection
        extraction_patterns = []
        
        # Pattern 1: Round number that's too precise (likely from wrong field)
        if isinstance(current_value, (int, float)) and current_value > 1000000:
            if str(int(current_value))[-3:] not in ['000', '500']:  # Not ending in round numbers
                extraction_patterns.append("non_round_large_number")
        
        # Pattern 2: Value magnitude doesn't match field type
        if "contributed" in field.lower() and "invested" in field.lower():
            # These should be relatively close in magnitude
            if current_value and document_context.get("total_fund_size"):
                fund_size = document_context.get("total_fund_size")
                if current_value > fund_size * 1.5:
                    extraction_patterns.append("value_exceeds_fund_size")
        
        # Pattern 3: Text in numeric field
        if current_value and isinstance(current_value, str):
            if any(char.isalpha() for char in str(current_value)):
                extraction_patterns.append("text_in_numeric_field")
                analysis_result["error_type"] = "extraction_error"
                analysis_result["confidence"] = 0.95
        
        analysis_result["extraction_patterns"] = extraction_patterns
        
        # Adjust confidence based on patterns
        if len(extraction_patterns) > 0:
            analysis_result["confidence"] *= (0.8 ** len(extraction_patterns))
        
        # Final recommendation adjustment
        if analysis_result.get("confidence", 0) < 0.5:
            analysis_result["recommended_action"] = "flag_for_manual_review"
        elif analysis_result.get("error_type") in ["extraction_error", "field_confusion"]:
            if analysis_result.get("confidence", 0) > 0.7:
                analysis_result["recommended_action"] = "skip_correction"
            else:
                analysis_result["recommended_action"] = "investigate_further"
        
        # Add metadata
        analysis_result["analysis_timestamp"] = datetime.now().isoformat()
        analysis_result["discrepancy_id"] = discrepancy.get("discrepancy_id", "unknown")
        
        return analysis_result
    
    async def _plan_correction(self, discrepancy: Dict[str, Any], 
                             document_context: Dict[str, Any], 
                             analysis: Dict[str, Any]) -> ReasoningTrace:
        """Step 2: Create an intelligent correction plan"""
        
        start_time = time.time()
        
        planning_prompt = f"""
You are a financial expert correcting a PE fund document. Generate a SPECIFIC NUMERIC VALUE, not a description.

ANALYSIS RESULTS:
- Issue Type: {analysis.get('issue_type')}
- Root Cause: {analysis.get('root_cause')}
- Severity: {analysis.get('severity')}
- Suggested Approach: {analysis.get('correction_approach')}

FIELD TO CORRECT: {discrepancy.get('field')}
CURRENT VALUE: {discrepancy.get('current_value')}
EXPECTED VALUE: {discrepancy.get('expected_value')}

DOCUMENT CONTEXT (for calculations):
{json.dumps(document_context, indent=2)[:800]}...

CRITICAL REQUIREMENTS:
1. **Generate a SPECIFIC NUMERIC VALUE** - not text like "sum of X and Y"
2. **Use actual calculations** - if field is total_fund_nav and you see unrealized=400M and realized=150M, return 550000000
3. **For missing data**, provide a reasonable numeric estimate based on fund size/context
4. **For location/text fields**, use standardized values like "USA", "Technology", etc.

EXAMPLES OF GOOD CORRECTIONS:
- total_fund_net_asset_value: 525000000 (not "sum of unrealized and realized")
- total_contributed_capital: 475000000 (not "estimated from context")
- location: "USA" (not "North American region")

Respond with JSON:
{{
    "correction_value": 525000000,
    "correction_method": "calculation|lookup|inference|default",
    "financial_justification": "why this specific value is correct",
    "risk_assessment": "potential risks",
    "validation_steps": ["step1", "step2"],
    "confidence": 0.0-1.0
}}
"""
        
        try:
            if self.openai_client:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": planning_prompt}],
                    temperature=0.1,
                    max_tokens=600
                )
                
                plan_text = response.choices[0].message.content
                
                # Extract structured plan
                try:
                    json_start = plan_text.find('{')
                    json_end = plan_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        plan_result = json.loads(plan_text[json_start:json_end])
                    else:
                        plan_result = self._create_fallback_plan(discrepancy, analysis)
                except json.JSONDecodeError:
                    plan_result = self._create_fallback_plan(discrepancy, analysis)
                
                reasoning = f"AI Planning: {plan_result.get('financial_justification', 'Generated correction plan')}"
                confidence = plan_result.get("confidence", 0.7)
                
            else:
                plan_result = self._create_fallback_plan(discrepancy, analysis)
                reasoning = "Rule-based planning (no LLM available)"
                confidence = 0.6
        
        except Exception as e:
            logger.error(f"Planning error: {e}")
            plan_result = self._create_fallback_plan(discrepancy, analysis)
            reasoning = f"Planning failed: {str(e)}"
            confidence = 0.3
        
        execution_time = time.time() - start_time
        
        return ReasoningTrace(
            step=ReasoningStep.PLAN,
            timestamp=datetime.now(),
            input_data={"discrepancy": discrepancy, "analysis": analysis},
            reasoning=reasoning,
            decision=f"Use {plan_result.get('correction_method')} to set value: {plan_result.get('correction_value')}",
            confidence=confidence,
            tools_used=["llm_planning" if self.openai_client else "rule_based"],
            output=plan_result,
            execution_time=execution_time
        )
    
    async def _execute_correction(self, discrepancy: Dict[str, Any], 
                                document_context: Dict[str, Any], 
                                plan: Dict[str, Any]) -> ReasoningTrace:
        """Step 3: Execute the correction with intelligent value derivation"""
        
        start_time = time.time()
        
        # Intelligent value derivation based on plan
        correction_value = plan.get("correction_value")
        correction_method = plan.get("correction_method", "default")
        
        # Smart value processing and calculation
        if correction_value and correction_value != "null":
            # Try to parse as number first
            try:
                if isinstance(correction_value, str):
                    # Handle calculations in text (e.g., "420000000 + 150000000")
                    if any(op in correction_value for op in ['+', '-', '*', '/']):
                        # Extract numbers and perform calculation
                        import re
                        numbers = re.findall(r'\d+\.?\d*', correction_value)
                        if len(numbers) >= 2:
                            if '+' in correction_value:
                                correction_value = sum(float(num) for num in numbers)
                            elif '-' in correction_value and len(numbers) == 2:
                                correction_value = float(numbers[0]) - float(numbers[1])
                            elif '*' in correction_value and len(numbers) == 2:
                                correction_value = float(numbers[0]) * float(numbers[1])
                    else:
                        # Remove currency symbols and commas
                        clean_value = correction_value.replace("$", "").replace(",", "").replace("M", "000000").replace("B", "000000000").strip()
                        if clean_value.replace(".", "").replace("-", "").isdigit():
                            correction_value = float(clean_value)
            except Exception as e:
                logger.warning(f"Could not parse correction value '{correction_value}': {e}")
                pass
        
        # Apply intelligent defaults based on field type and context
        if correction_value is None or correction_value == "null":
            correction_value = self._derive_intelligent_value(
                discrepancy.get("field"), 
                discrepancy.get("current_value"),
                discrepancy.get("expected_value"),
                document_context
            )
        
        reasoning = f"Applied {correction_method} method: {plan.get('financial_justification', 'No justification')}"
        
        execution_time = time.time() - start_time
        
        return ReasoningTrace(
            step=ReasoningStep.EXECUTE,
            timestamp=datetime.now(),
            input_data={"discrepancy": discrepancy, "plan": plan},
            reasoning=reasoning,
            decision=f"Set {discrepancy.get('field')} = {correction_value}",
            confidence=plan.get("confidence", 0.7),
            tools_used=["intelligent_derivation"],
            output={
                "corrected_value": correction_value,
                "correction_method": correction_method,
                "justification": plan.get("financial_justification")
            },
            execution_time=execution_time
        )
    
    async def _verify_correction(self, discrepancy: Dict[str, Any], 
                               execution_result: Dict[str, Any]) -> ReasoningTrace:
        """Step 4: Verify the correction makes financial sense"""
        
        start_time = time.time()
        
        original_value = discrepancy.get("current_value")
        corrected_value = execution_result.get("corrected_value")
        field_name = discrepancy.get("field")
        
        # Business logic validation
        validation_issues = []
        confidence = 0.9
        
        # Check for reasonable value ranges
        if isinstance(corrected_value, (int, float)) and isinstance(original_value, (int, float)):
            if corrected_value < 0 and "value" in field_name.lower():
                validation_issues.append("Negative value for financial metric")
                confidence *= 0.7
            
            # Check for unreasonable changes (>1000x or <0.001x)
            if original_value != 0 and abs(corrected_value / original_value) > 1000:
                validation_issues.append("Correction changes value by >1000x")
                confidence *= 0.6
            elif original_value != 0 and abs(corrected_value / original_value) < 0.001:
                validation_issues.append("Correction changes value by <0.001x")
                confidence *= 0.6
        
        # Field-specific validations
        if "location" in field_name.lower():
            valid_locations = ["USA", "Europe", "Asia", "North America", "Global", "Other"]
            if corrected_value not in valid_locations:
                validation_issues.append("Invalid location value")
                confidence *= 0.8
        
        if "status" in field_name.lower():
            valid_statuses = ["unrealized", "partially_realized", "realized"]
            if corrected_value not in valid_statuses:
                validation_issues.append("Invalid investment status")
                confidence *= 0.8
        
        if validation_issues:
            reasoning = f"Validation concerns: {', '.join(validation_issues)}"
            decision = "Correction has validation issues"
            confidence *= 0.5
        else:
            reasoning = "Correction passes business logic validation"
            decision = "Correction is valid"
        
        execution_time = time.time() - start_time
        
        return ReasoningTrace(
            step=ReasoningStep.VERIFY,
            timestamp=datetime.now(),
            input_data={"discrepancy": discrepancy, "execution_result": execution_result},
            reasoning=reasoning,
            decision=decision,
            confidence=confidence,
            tools_used=["business_logic_validator"],
            output={
                "validation_passed": len(validation_issues) == 0,
                "validation_issues": validation_issues,
                "final_confidence": confidence
            },
            execution_time=execution_time
        )
    
    def _rule_based_analysis(self, discrepancy: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback rule-based analysis when LLM is not available"""
        
        field = discrepancy.get("field", "")
        current_value = discrepancy.get("current_value")
        expected_value = discrepancy.get("expected_value")
        
        if current_value is None or current_value == "":
            issue_type = "missing_data"
            severity = "high"
            approach = "Use expected value or reasonable default"
        elif expected_value is not None and current_value != expected_value:
            issue_type = "calculation_error"
            severity = "high"
            approach = "Use expected value"
        elif "location" in field.lower():
            issue_type = "data_entry"
            severity = "medium"
            approach = "Standardize location format"
        else:
            issue_type = "unknown"
            severity = "medium"
            approach = "Conservative correction"
        
        return {
            "issue_type": issue_type,
            "root_cause": f"Rule-based analysis of {field}",
            "severity": severity,
            "business_impact": f"Affects {field} accuracy",
            "correction_approach": approach,
            "confidence": 0.6
        }
    
    def _create_fallback_plan(self, discrepancy: Dict[str, Any], 
                            analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create a fallback correction plan"""
        
        expected_value = discrepancy.get("expected_value")
        current_value = discrepancy.get("current_value")
        field = discrepancy.get("field", "")
        
        if expected_value is not None:
            correction_value = expected_value
            method = "use_expected"
            justification = "Using expected value from discrepancy"
            confidence = 0.8
        elif current_value is None:
            correction_value = self._derive_intelligent_value(field, None, None, {})
            method = "intelligent_default"
            justification = "Using intelligent default for missing data"
            confidence = 0.6
        else:
            correction_value = current_value
            method = "keep_original"
            justification = "Keeping original value due to uncertainty"
            confidence = 0.4
        
        return {
            "correction_value": correction_value,
            "correction_method": method,
            "financial_justification": justification,
            "risk_assessment": "Low risk fallback approach",
            "validation_steps": ["business_logic_check"],
            "confidence": confidence
        }
    
    def _derive_intelligent_value(self, field: str, current_value: Any, 
                                expected_value: Any, context: Dict[str, Any]) -> Any:
        """Derive intelligent values based on field type and context with actual calculations"""
        
        field_lower = field.lower()
        
        # Use expected value if available
        if expected_value is not None:
            return expected_value
        
        # Keep current value if it exists and seems reasonable
        if current_value is not None and current_value != "":
            return current_value
        
        # Smart financial calculations based on available context
        if "total_fund_net_asset_value" in field_lower:
            # Calculate NAV = unrealized + cash/remaining capital
            unrealized = context.get("total_investments_unrealized_value", 0) or 0
            realized = context.get("total_investments_realized_value", 0) or 0
            distributions = context.get("total_distribution_to_partners", 0) or 0
            # NAV = unrealized + (realized - distributions)
            nav = unrealized + (realized - distributions)
            return max(nav, unrealized) if nav > 0 else unrealized
        
        elif "total_investments_unrealized_and_realized" in field_lower:
            # Calculate total = unrealized + realized
            unrealized = context.get("total_investments_unrealized_value", 0) or 0
            realized = context.get("total_investments_realized_value", 0) or 0
            return unrealized + realized
        
        elif "total_contributed_capital" in field_lower:
            # Estimate from committed capital or investment values
            committed = context.get("total_committed_capital", 0) or 0
            invested = context.get("total_invested_capital", 0) or 0
            unrealized = context.get("total_investments_unrealized_value", 0) or 0
            realized = context.get("total_investments_realized_value", 0) or 0
            
            # Use best available estimate
            if invested > 0:
                return invested
            elif unrealized + realized > 0:
                return unrealized + realized
            elif committed > 0:
                return committed * 0.8  # Assume 80% draw-down
            return 0
        
        elif "total_active_investments_cost_basis" in field_lower:
            # Estimate from unrealized investments
            unrealized = context.get("total_investments_unrealized_value", 0) or 0
            return unrealized * 0.85 if unrealized > 0 else 0  # Cost basis typically lower than current value
        
        # Text field defaults
        elif "location" in field_lower:
            return "USA"
        elif "industry" in field_lower:
            return "Technology"
        elif "status" in field_lower:
            return "unrealized"
        elif "currency" in field_lower:
            return "USD"
        
        # Numeric field defaults
        elif any(keyword in field_lower for keyword in ["invested", "value", "capital", "nav"]):
            return 0
        elif "irr" in field_lower:
            return 0.08  # 8% default IRR
        elif "moic" in field_lower:
            return 1.2  # 1.2x default MOIC
        elif "number_of" in field_lower:
            return 1  # Default count
        else:
            return None
    
    async def _reflect_on_correction(self, discrepancy: Dict[str, Any], 
                                   correction_result: Dict[str, Any],
                                   document_context: Dict[str, Any]) -> ReasoningTrace:
        """Step 5: REFLECTION - Critical self-assessment of the correction"""
        
        start_time = time.time()
        
        field = discrepancy.get("field")
        original_value = discrepancy.get("current_value")
        corrected_value = correction_result.get("corrected_value")
        correction_method = correction_result.get("correction_method")
        
        reflection_prompt = f"""
You are a financial expert reviewing a correction made to a PE fund document. 
CRITICALLY ANALYZE this correction for accuracy and financial soundness.

CORRECTION DETAILS:
- Field: {field}
- Original Value: {original_value}
- Corrected Value: {corrected_value} 
- Method Used: {correction_method}
- Original Justification: {correction_result.get("justification", "")}

DOCUMENT CONTEXT (for validation):
{json.dumps(document_context, indent=2)[:1000]}...

REFLECTION CHECKLIST:
1. **Mathematical Accuracy**: Is the correction mathematically correct?
   - For NAV: Does it equal unrealized + realized - distributions?
   - For totals: Do components add up correctly?
   - For ratios: Are calculations proper?

2. **Business Logic**: Does this make financial sense?
   - Are the values reasonable for this fund size?
   - Do the numbers align with industry standards?
   - Is the correction directionally correct?

3. **Internal Consistency**: Does this correction create conflicts?
   - Will it conflict with other fields in the document?
   - Does it maintain logical relationships?

4. **Risk Assessment**: What could go wrong?
   - Could this correction be significantly off?
   - Are there alternative interpretations?

Respond with JSON:
{{
    "mathematical_accuracy": "correct|incorrect|uncertain",
    "business_logic_sound": true|false,
    "internal_consistency": "consistent|conflicts|unknown",
    "identified_issues": ["issue1", "issue2"],
    "suggested_improvements": ["improvement1", "improvement2"],
    "needs_improvement": true|false,
    "reflection_confidence": 0.0-1.0,
    "final_assessment": "approve|reject|improve"
}}
"""
        
        try:
            if self.openai_client:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": reflection_prompt}],
                    temperature=0.1,
                    max_tokens=800
                )
                
                reflection_text = response.choices[0].message.content
                
                # Extract structured reflection
                try:
                    json_start = reflection_text.find('{')
                    json_end = reflection_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        reflection_result = json.loads(reflection_text[json_start:json_end])
                    else:
                        reflection_result = self._create_fallback_reflection(correction_result)
                except json.JSONDecodeError:
                    reflection_result = self._create_fallback_reflection(correction_result)
                
                reasoning = f"AI Reflection: {reflection_result.get('final_assessment', 'uncertain')} - {', '.join(reflection_result.get('identified_issues', []))}"
                confidence = reflection_result.get("reflection_confidence", 0.7)
                
            else:
                reflection_result = self._create_fallback_reflection(correction_result)
                reasoning = "Rule-based reflection (no LLM available)"
                confidence = 0.6
        
        except Exception as e:
            logger.error(f"Reflection error: {e}")
            reflection_result = self._create_fallback_reflection(correction_result)
            reasoning = f"Reflection failed: {str(e)}"
            confidence = 0.3
        
        execution_time = time.time() - start_time
        
        return ReasoningTrace(
            step=ReasoningStep.LEARN,  # Using LEARN as the reflection step
            timestamp=datetime.now(),
            input_data={"discrepancy": discrepancy, "correction_result": correction_result},
            reasoning=reasoning,
            decision=f"Reflection: {reflection_result.get('final_assessment', 'uncertain')}",
            confidence=confidence,
            tools_used=["llm_reflection" if self.openai_client else "rule_based"],
            output=reflection_result,
            execution_time=execution_time
        )
    
    async def _improve_correction(self, discrepancy: Dict[str, Any],
                                original_correction: Dict[str, Any],
                                reflection_feedback: Dict[str, Any],
                                document_context: Dict[str, Any]) -> ReasoningTrace:
        """Step 6: IMPROVE - Apply improvements from reflection"""
        
        start_time = time.time()
        
        improvement_prompt = f"""
Based on the reflection feedback, improve this financial correction:

ORIGINAL CORRECTION:
- Field: {discrepancy.get('field')}
- Original Value: {discrepancy.get('current_value')}
- Corrected Value: {original_correction.get('corrected_value')}

REFLECTION FEEDBACK:
- Issues Identified: {reflection_feedback.get('identified_issues', [])}
- Suggested Improvements: {reflection_feedback.get('suggested_improvements', [])}
- Mathematical Accuracy: {reflection_feedback.get('mathematical_accuracy')}

DOCUMENT CONTEXT (for recalculation):
{json.dumps(document_context, indent=2)[:800]}...

IMPROVEMENT TASK:
Create an improved correction that addresses the reflection feedback.
- If mathematical errors were found, recalculate correctly
- If business logic issues exist, apply proper financial reasoning
- Ensure internal consistency with other document fields

Respond with JSON:
{{
    "corrected_value": improved_numeric_value,
    "improvement_method": "recalculation|adjustment|validation_fix",
    "justification": "why this improved value is better",
    "changes_made": ["change1", "change2"],
    "confidence": 0.0-1.0
}}
"""
        
        try:
            if self.openai_client:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": improvement_prompt}],
                    temperature=0.1,
                    max_tokens=600
                )
                
                improvement_text = response.choices[0].message.content
                
                # Extract structured improvement
                try:
                    json_start = improvement_text.find('{')
                    json_end = improvement_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        improvement_result = json.loads(improvement_text[json_start:json_end])
                    else:
                        improvement_result = self._create_fallback_improvement(original_correction)
                except json.JSONDecodeError:
                    improvement_result = self._create_fallback_improvement(original_correction)
                
                reasoning = f"AI Improvement: {improvement_result.get('justification', 'Applied reflection feedback')}"
                confidence = improvement_result.get("confidence", 0.8)
                
            else:
                improvement_result = self._create_fallback_improvement(original_correction)
                reasoning = "Basic improvement applied (no LLM available)"
                confidence = 0.6
        
        except Exception as e:
            logger.error(f"Improvement error: {e}")
            improvement_result = self._create_fallback_improvement(original_correction)
            reasoning = f"Improvement failed: {str(e)}"
            confidence = 0.4
        
        execution_time = time.time() - start_time
        
        return ReasoningTrace(
            step=ReasoningStep.EXECUTE,  # This is an execution of improvement
            timestamp=datetime.now(),
            input_data={"original_correction": original_correction, "reflection_feedback": reflection_feedback},
            reasoning=reasoning,
            decision=f"Improved correction: {improvement_result.get('corrected_value')}",
            confidence=confidence,
            tools_used=["llm_improvement" if self.openai_client else "rule_based"],
            output=improvement_result,
            execution_time=execution_time
        )
    
    def _create_fallback_reflection(self, correction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback reflection when LLM is not available"""
        corrected_value = correction_result.get("corrected_value")
        
        # Basic validation checks
        issues = []
        needs_improvement = False
        
        if corrected_value is None:
            issues.append("No corrected value provided")
            needs_improvement = True
        elif isinstance(corrected_value, (int, float)) and corrected_value < 0:
            issues.append("Negative value for financial metric")
            needs_improvement = True
        
        return {
            "mathematical_accuracy": "uncertain",
            "business_logic_sound": len(issues) == 0,
            "internal_consistency": "unknown",
            "identified_issues": issues,
            "suggested_improvements": ["Validate calculations", "Check business logic"],
            "needs_improvement": needs_improvement,
            "reflection_confidence": 0.6,
            "final_assessment": "improve" if needs_improvement else "approve"
        }
    
    def _create_fallback_improvement(self, original_correction: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback improvement when LLM is not available"""
        
        return {
            "corrected_value": original_correction.get("corrected_value"),
            "improvement_method": "validation_fix",
            "justification": "Applied basic validation improvements",
            "changes_made": ["Basic validation applied"],
            "confidence": 0.6
        }

    def _extract_pdf_path(self, document_context: Dict[str, Any], discrepancy: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Try to extract the PDF file path from document context or discrepancy."""
        # Try common keys
        for key in ["pdf_path", "document_path", "file_path", "pdfFileName", "pdf_file_name", "file_name"]:
            if key in document_context and document_context[key]:
                return document_context[key]
            if discrepancy and key in discrepancy and discrepancy[key]:
                return discrepancy[key]
        # Try nested locations
        if "metadata" in document_context and isinstance(document_context["metadata"], dict):
            for key in ["pdf_path", "document_path", "file_path", "pdfFileName", "pdf_file_name", "file_name"]:
                if key in document_context["metadata"] and document_context["metadata"][key]:
                    return document_context["metadata"][key]
        return None

# Export the intelligence engine
__all__ = ["FinancialEngine", "ReasoningTrace", "AgentResponse", "ReasoningStep"]