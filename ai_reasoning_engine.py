"""
AI Reasoning Engine - Core Intelligence for Financial Document Processing
Google-level AI agent with conversational interface, multi-step reasoning, and tool use
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
    Implements chain-of-thought reasoning, tool use, and conversational interface
    """
    
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.reasoning_history = []
        self.conversation_context = []
        self.available_tools = {}
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
        
        # Determine final decision based on reasoning chain - fix the list access bug
        final_confidence = min(trace.confidence for trace in reasoning_chain if hasattr(trace, 'confidence'))
        
        if final_confidence >= 0.7 and execution_trace.output.get("corrected_value") is not None:  # Lower threshold
            success = True
            message = f"Successfully corrected {discrepancy.get('field')} with {final_confidence:.1%} confidence"
            final_decision = {
                "action": "correct",
                "field": discrepancy.get("field"),
                "original_value": discrepancy.get("current_value"),
                "corrected_value": execution_trace.output.get("corrected_value"),
                "reasoning": execution_trace.reasoning,
                "confidence": final_confidence
            }
        else:
            success = False
            message = f"Could not correct {discrepancy.get('field')} - insufficient confidence or unclear solution"
            final_decision = {
                "action": "flag_for_review",
                "field": discrepancy.get("field"),
                "reason": verification_trace.reasoning,
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
    
    async def _plan_correction(self, discrepancy: Dict[str, Any], 
                             document_context: Dict[str, Any], 
                             analysis: Dict[str, Any]) -> ReasoningTrace:
        """Step 2: Create an intelligent correction plan"""
        
        start_time = time.time()
        
        planning_prompt = f"""
Based on the analysis, create a specific correction plan:

ANALYSIS RESULTS:
- Issue Type: {analysis.get('issue_type')}
- Root Cause: {analysis.get('root_cause')}
- Severity: {analysis.get('severity')}
- Suggested Approach: {analysis.get('correction_approach')}

FIELD TO CORRECT: {discrepancy.get('field')}
CURRENT VALUE: {discrepancy.get('current_value')}
EXPECTED VALUE: {discrepancy.get('expected_value')}

PLANNING TASK:
Create a specific, actionable correction plan. Consider:
1. **What value should we use?** Be specific
2. **Why is this the right choice?** Financial reasoning
3. **What are the risks?** What could go wrong
4. **Validation checks**: How to verify the correction

Respond with:
{{
    "correction_value": "specific value to use",
    "correction_method": "calculation|lookup|inference|default",
    "financial_justification": "why this value makes financial sense",
    "risk_assessment": "potential risks of this correction",
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
        
        # Smart value processing
        if correction_value and correction_value != "null":
            # Try to parse as number if it looks like one
            try:
                if isinstance(correction_value, str):
                    # Remove currency symbols and commas
                    clean_value = correction_value.replace("$", "").replace(",", "").strip()
                    if clean_value.replace(".", "").replace("-", "").isdigit():
                        correction_value = float(clean_value)
            except:
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
        """Derive intelligent values based on field type and context"""
        
        field_lower = field.lower()
        
        # Use expected value if available
        if expected_value is not None:
            return expected_value
        
        # Keep current value if it exists and seems reasonable
        if current_value is not None and current_value != "":
            return current_value
        
        # Intelligent defaults based on field type
        if "location" in field_lower:
            return "USA"  # Most common location for US PE funds
        elif "industry" in field_lower:
            return "Technology"  # Most common industry
        elif "status" in field_lower:
            return "unrealized"  # Most common status
        elif "total_invested" in field_lower:
            # Try to derive from other values in context
            asset_data = context.get("assets", {})
            if isinstance(asset_data, dict) and field.startswith("assets."):
                asset_name = field.split(".")[1]
                asset_info = asset_data.get(asset_name, {})
                total_value = asset_info.get("total_value", 0)
                # Estimate invested as 80% of total value for unrealized assets
                return total_value * 0.8 if total_value else 0
            return 0
        elif "value" in field_lower:
            return 0
        else:
            return None

# Export the intelligence engine
__all__ = ["FinancialEngine", "ReasoningTrace", "AgentResponse", "ReasoningStep"]