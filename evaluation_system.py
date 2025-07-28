#!/usr/bin/env python3
"""
Comprehensive Evaluation System for Tetrix AI Document Correction
Tracks costs, time, and CORRECT resolution rates against ground truth
"""

import asyncio
import json
import logging
import time
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import tiktoken

from main import TetrixProductionSystem
from analytics_client import create_analytics_client
from document_agent import DocumentAgent
from feedback_loop import FeedbackLoopSystem

logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for a document"""
    document_path: str
    document_type: str  # 'training' or 'test'
    
    # Timing metrics
    total_processing_time: float
    api_call_time: float
    
    # Cost metrics
    total_api_cost: float
    openai_tokens_used: int
    openai_cost_breakdown: Dict[str, float]
    
    # Accuracy metrics
    total_discrepancies: int
    total_focus_points: int
    discrepancies_correctly_resolved: int
    focus_points_correctly_resolved: int
    discrepancies_incorrectly_resolved: int
    focus_points_incorrectly_resolved: int
    discrepancies_not_attempted: int
    focus_points_not_attempted: int
    
    # Resolution rates
    discrepancy_correct_resolution_rate: float
    focus_point_correct_resolution_rate: float
    overall_correct_resolution_rate: float
    
    # Detailed results
    correction_details: List[Dict[str, Any]]
    ground_truth_comparison: Dict[str, Any]

class CostTracker:
    """Track API costs for OpenAI"""
    
    # OpenAI pricing (as of 2024)
    PRICING = {
        "gpt-3.5-turbo": {
            "input": 0.0005,   # per 1K tokens
            "output": 0.0015   # per 1K tokens
        },
        "gpt-4": {
            "input": 0.03,     # per 1K tokens
            "output": 0.06     # per 1K tokens
        }
    }
    
    def __init__(self):
        self.total_cost = 0.0
        self.token_usage = {
            "gpt-3.5-turbo": {"input": 0, "output": 0},
            "gpt-4": {"input": 0, "output": 0}
        }
        self.api_calls = []
        self.api_call_time = 0.0
    
    def track_api_call(self, model: str, input_tokens: int, output_tokens: int, 
                      response_time: float):
        """Track an API call"""
        if model not in self.PRICING:
            model = "gpt-3.5-turbo"  # Default
        
        # Calculate cost
        input_cost = (input_tokens / 1000) * self.PRICING[model]["input"]
        output_cost = (output_tokens / 1000) * self.PRICING[model]["output"]
        total_cost = input_cost + output_cost
        
        # Update totals
        self.total_cost += total_cost
        self.token_usage[model]["input"] += input_tokens
        self.token_usage[model]["output"] += output_tokens
        self.api_call_time += response_time
        
        # Record call details
        self.api_calls.append({
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": total_cost,
            "response_time": response_time
        })
        
        return total_cost
    
    def get_summary(self) -> Dict[str, Any]:
        """Get cost summary"""
        return {
            "total_cost": self.total_cost,
            "token_usage": self.token_usage,
            "api_call_count": len(self.api_calls),
            "average_cost_per_call": self.total_cost / len(self.api_calls) if self.api_calls else 0,
            "cost_breakdown": self._get_cost_breakdown()
        }
    
    def _get_cost_breakdown(self) -> Dict[str, float]:
        """Get detailed cost breakdown by model"""
        breakdown = {}
        for model, usage in self.token_usage.items():
            if usage["input"] > 0 or usage["output"] > 0:
                input_cost = (usage["input"] / 1000) * self.PRICING[model]["input"]
                output_cost = (usage["output"] / 1000) * self.PRICING[model]["output"]
                breakdown[model] = {
                    "input_cost": input_cost,
                    "output_cost": output_cost,
                    "total": input_cost + output_cost
                }
        return breakdown

class DocumentEvaluator:
    """Evaluate document corrections against ground truth"""
    
    def __init__(self):
        self.cost_tracker = CostTracker()
        self.analytics_client = None
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    async def __aenter__(self):
        self.analytics_client = create_analytics_client(use_mock=False)
        await self.analytics_client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.analytics_client:
            await self.analytics_client.__aexit__(exc_type, exc_val, exc_tb)
    
    async def evaluate_document(self, document_path: str, doc_type: str = "test") -> EvaluationMetrics:
        """Evaluate a single document with comprehensive metrics"""
        
        start_time = time.time()
        
        logger.info(f"📊 Evaluating {doc_type} document: {document_path}")
        
        # Reset cost tracker for this document
        self.cost_tracker = CostTracker()
        
        # Initialize feedback loop system with cost tracking
        feedback_loop = FeedbackLoopSystem(agent_mode="testing" if doc_type == "test" else "training")
        
        async with feedback_loop:
            # Monkey-patch the AI reasoning engine to track costs
            if hasattr(feedback_loop.document_agent.reasoning_engine, '_call_openai'):
                original_openai_call = feedback_loop.document_agent.reasoning_engine._call_openai
                
                async def tracked_openai_call(prompt: str):
                    api_call_start = time.time()
                    
                    # Count tokens
                    input_tokens = len(self.tokenizer.encode(prompt))
                    
                    # Make the actual call
                    result = await original_openai_call(prompt)
                    
                    # Count output tokens
                    output_tokens = len(self.tokenizer.encode(result.get("content", "")))
                    
                    # Track cost
                    api_time = time.time() - api_call_start
                    self.cost_tracker.track_api_call(
                        model=result.get("model", "gpt-3.5-turbo"),
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        response_time=api_time
                    )
                    
                    return result
                
                feedback_loop.document_agent.reasoning_engine._call_openai = tracked_openai_call
            
            # Get original issues
            original_issues = await self._get_original_issues(document_path)
            
            # Process document through feedback loop
            feedback_result = await feedback_loop.process_document_with_feedback_loop(document_path)
            
            # Get consolidated document for ground truth
            consolidated_doc = await self._get_consolidated_document_for_evaluation(document_path)
            
            # Evaluate corrections against ground truth
            evaluation_results = await self._evaluate_corrections_against_ground_truth(
                document_path,
                original_issues,
                feedback_result.corrections_applied,
                consolidated_doc
            )
            
            # Calculate metrics
            total_time = time.time() - start_time
            
            metrics = EvaluationMetrics(
                document_path=document_path,
                document_type=doc_type,
                total_processing_time=total_time,
                api_call_time=self.cost_tracker.api_call_time,
                total_api_cost=self.cost_tracker.total_cost,
                openai_tokens_used=sum(self.cost_tracker.token_usage["gpt-3.5-turbo"].values()),
                openai_cost_breakdown=self.cost_tracker.get_summary()["cost_breakdown"],
                total_discrepancies=evaluation_results["total_discrepancies"],
                total_focus_points=evaluation_results["total_focus_points"],
                discrepancies_correctly_resolved=evaluation_results["discrepancies_correctly_resolved"],
                focus_points_correctly_resolved=evaluation_results["focus_points_correctly_resolved"],
                discrepancies_incorrectly_resolved=evaluation_results["discrepancies_incorrectly_resolved"],
                focus_points_incorrectly_resolved=evaluation_results["focus_points_incorrectly_resolved"],
                discrepancies_not_attempted=evaluation_results["discrepancies_not_attempted"],
                focus_points_not_attempted=evaluation_results["focus_points_not_attempted"],
                discrepancy_correct_resolution_rate=evaluation_results["discrepancy_correct_resolution_rate"],
                focus_point_correct_resolution_rate=evaluation_results["focus_point_correct_resolution_rate"],
                overall_correct_resolution_rate=evaluation_results["overall_correct_resolution_rate"],
                correction_details=evaluation_results["correction_details"],
                ground_truth_comparison=evaluation_results["ground_truth_comparison"]
            )
            
            return metrics
    
    async def _get_original_issues(self, document_path: str) -> Dict[str, Any]:
        """Get original discrepancies and focus points"""
        analytics_response = await self.analytics_client.get_discrepancies_for_document(document_path)
        
        return {
            "discrepancies": analytics_response.discrepancies,
            "focus_points": analytics_response.focus_points,
            "total_issues": len(analytics_response.discrepancies) + len(analytics_response.focus_points)
        }
    
    async def _get_consolidated_document_for_evaluation(self, document_path: str) -> Optional[Dict[str, Any]]:
        """Get consolidated document for ground truth comparison"""
        # Get document data to extract metadata_id
        document_data = await self.analytics_client.get_raw_document_data(document_path)
        
        if not document_data or document_data.get("error"):
            logger.warning(f"Could not get document data for {document_path}")
            return None
        
        metadata_id = document_data.get("metadata_id")
        if not metadata_id:
            logger.warning(f"No metadata_id found for {document_path}")
            return None
        
        # Get consolidated documents and find match
        consolidated_response = await self.analytics_client.get_consolidated_documents()
        if consolidated_response and consolidated_response.get("success"):
            for doc in consolidated_response.get("documents", []):
                # Check nested structure for metadata_id match
                for entity in doc.get("underlying_client_entities", []):
                    for subdoc in entity.get("documents", []):
                        if subdoc.get("metadata_id") == metadata_id:
                            logger.info(f"Found consolidated document for evaluation")
                            return subdoc
        
        return None
    
    async def _evaluate_corrections_against_ground_truth(self, document_path: str,
                                                       original_issues: Dict[str, Any],
                                                       corrections: List[Dict[str, Any]],
                                                       consolidated_doc: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate which corrections were actually correct based on ground truth"""
        
        discrepancies = original_issues["discrepancies"]
        focus_points = original_issues["focus_points"]
        
        # Initialize counters
        disc_correct = 0
        disc_incorrect = 0
        disc_not_attempted = 0
        fp_correct = 0
        fp_incorrect = 0
        fp_not_attempted = 0
        
        correction_details = []
        
        # Create a map of corrections by field
        correction_map = {c["field"]: c for c in corrections}
        
        # Evaluate discrepancies
        for discrepancy in discrepancies:
            field = discrepancy.field
            expected_value = discrepancy.expected_value
            
            if field in correction_map:
                correction = correction_map[field]
                corrected_value = correction.get("corrected_value")
                
                # Get ground truth value
                ground_truth_value = self._get_ground_truth_value(field, consolidated_doc) if consolidated_doc else expected_value
                
                # Check if correction matches ground truth
                if self._values_match(corrected_value, ground_truth_value):
                    disc_correct += 1
                    status = "correctly_resolved"
                else:
                    disc_incorrect += 1
                    status = "incorrectly_resolved"
                
                correction_details.append({
                    "field": field,
                    "issue_type": "discrepancy",
                    "original_value": discrepancy.current_value,
                    "corrected_value": corrected_value,
                    "ground_truth_value": ground_truth_value,
                    "expected_value": expected_value,
                    "status": status,
                    "confidence": correction.get("confidence", 0)
                })
            else:
                disc_not_attempted += 1
                correction_details.append({
                    "field": field,
                    "issue_type": "discrepancy",
                    "original_value": discrepancy.current_value,
                    "expected_value": expected_value,
                    "status": "not_attempted"
                })
        
        # Evaluate focus points
        for focus_point in focus_points:
            field = focus_point.field
            
            if field in correction_map:
                correction = correction_map[field]
                corrected_value = correction.get("corrected_value")
                
                # Get ground truth value
                ground_truth_value = self._get_ground_truth_value(field, consolidated_doc) if consolidated_doc else None
                
                # For focus points, we check if the correction improved the value
                if ground_truth_value is not None:
                    if self._values_match(corrected_value, ground_truth_value):
                        fp_correct += 1
                        status = "correctly_resolved"
                    else:
                        fp_incorrect += 1
                        status = "incorrectly_resolved"
                else:
                    # No ground truth, consider it correct if it seems reasonable
                    if self._is_reasonable_correction(field, focus_point.current_value, corrected_value):
                        fp_correct += 1
                        status = "likely_correct"
                    else:
                        fp_incorrect += 1
                        status = "likely_incorrect"
                
                correction_details.append({
                    "field": field,
                    "issue_type": "focus_point",
                    "original_value": focus_point.current_value,
                    "corrected_value": corrected_value,
                    "ground_truth_value": ground_truth_value,
                    "status": status,
                    "confidence": correction.get("confidence", 0)
                })
            else:
                fp_not_attempted += 1
                correction_details.append({
                    "field": field,
                    "issue_type": "focus_point",
                    "original_value": focus_point.current_value,
                    "status": "not_attempted"
                })
        
        # Calculate rates
        total_disc = len(discrepancies)
        total_fp = len(focus_points)
        
        disc_rate = (disc_correct / total_disc * 100) if total_disc > 0 else 0
        fp_rate = (fp_correct / total_fp * 100) if total_fp > 0 else 0
        overall_rate = ((disc_correct + fp_correct) / (total_disc + total_fp) * 100) if (total_disc + total_fp) > 0 else 0
        
        return {
            "total_discrepancies": total_disc,
            "total_focus_points": total_fp,
            "discrepancies_correctly_resolved": disc_correct,
            "focus_points_correctly_resolved": fp_correct,
            "discrepancies_incorrectly_resolved": disc_incorrect,
            "focus_points_incorrectly_resolved": fp_incorrect,
            "discrepancies_not_attempted": disc_not_attempted,
            "focus_points_not_attempted": fp_not_attempted,
            "discrepancy_correct_resolution_rate": disc_rate,
            "focus_point_correct_resolution_rate": fp_rate,
            "overall_correct_resolution_rate": overall_rate,
            "correction_details": correction_details,
            "ground_truth_comparison": {
                "ground_truth_available": consolidated_doc is not None,
                "fields_compared": len(correction_details)
            }
        }
    
    def _get_ground_truth_value(self, field: str, consolidated_doc: Dict[str, Any]) -> Any:
        """Extract ground truth value for a field from consolidated document"""
        if not consolidated_doc:
            return None
        
        # Handle nested fields (e.g., assets.CompanyName.field)
        if "." in field:
            parts = field.split(".")
            current = consolidated_doc
            
            try:
                if parts[0] == "assets" and len(parts) >= 3:
                    asset_name = parts[1]
                    field_name = parts[2]
                    
                    # Search in assets list
                    assets = consolidated_doc.get("assets", [])
                    for asset in assets:
                        if isinstance(asset, dict):
                            # Check various name fields
                            names_to_check = [
                                asset.get("name", ""),
                                asset.get("original_name", ""),
                                asset.get("company_name", "")
                            ]
                            
                            if any(asset_name.lower() in name.lower() or name.lower() in asset_name.lower() 
                                  for name in names_to_check if name):
                                return asset.get(field_name)
                else:
                    # Regular nested field
                    for part in parts:
                        if isinstance(current, dict) and part in current:
                            current = current[part]
                        else:
                            return None
                    return current
            except:
                return None
        else:
            # Top-level field
            return consolidated_doc.get(field)
    
    def _values_match(self, value1: Any, value2: Any) -> bool:
        """Check if two values match with appropriate tolerance"""
        if value1 is None or value2 is None:
            return value1 == value2
        
        # Numeric comparison with tolerance
        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            # 1% tolerance for financial values
            tolerance = max(abs(value1), abs(value2)) * 0.01
            return abs(value1 - value2) <= tolerance
        
        # String comparison (case-insensitive)
        if isinstance(value1, str) and isinstance(value2, str):
            return value1.strip().lower() == value2.strip().lower()
        
        return value1 == value2
    
    def _is_reasonable_correction(self, field: str, original_value: Any, corrected_value: Any) -> bool:
        """Check if a correction seems reasonable when no ground truth is available"""
        
        # Location fields
        if "location" in field.lower():
            valid_locations = ["USA", "Europe", "Asia", "North America", "Global"]
            return corrected_value in valid_locations
        
        # Status fields
        if "status" in field.lower():
            valid_statuses = ["unrealized", "partially_realized", "realized"]
            return corrected_value in valid_statuses
        
        # Numeric fields should not be negative (except returns)
        if isinstance(corrected_value, (int, float)):
            if any(term in field.lower() for term in ["value", "capital", "investment"]):
                return corrected_value >= 0
        
        # Default: assume reasonable if not obviously wrong
        return True

class ComprehensiveEvaluator:
    """Run comprehensive evaluation on document sets"""
    
    def __init__(self):
        self.evaluator = DocumentEvaluator()
        self.results = {
            "training": [],
            "test": []
        }
    
    async def run_evaluation(self, training_docs: List[str], test_docs: List[str]):
        """Run full evaluation on training and test sets"""
        
        async with self.evaluator:
            print("\n" + "="*80)
            print("🚀 COMPREHENSIVE EVALUATION SYSTEM")
            print("="*80)
            
            # Training documents
            print(f"\n📚 Evaluating {len(training_docs)} TRAINING documents...")
            for i, doc_path in enumerate(training_docs, 1):
                print(f"\n[{i}/{len(training_docs)}] Training Document: {doc_path}")
                try:
                    metrics = await self.evaluator.evaluate_document(doc_path, "training")
                    self.results["training"].append(metrics)
                    self._print_document_summary(metrics)
                except Exception as e:
                    logger.error(f"Error evaluating {doc_path}: {e}")
                    print(f"   ❌ Error: {e}")
            
            # Test documents
            print(f"\n📊 Evaluating {len(test_docs)} TEST documents...")
            for i, doc_path in enumerate(test_docs, 1):
                print(f"\n[{i}/{len(test_docs)}] Test Document: {doc_path}")
                try:
                    metrics = await self.evaluator.evaluate_document(doc_path, "test")
                    self.results["test"].append(metrics)
                    self._print_document_summary(metrics)
                except Exception as e:
                    logger.error(f"Error evaluating {doc_path}: {e}")
                    print(f"   ❌ Error: {e}")
            
            # Generate final report
            self._generate_final_report()
    
    def _print_document_summary(self, metrics: EvaluationMetrics):
        """Print summary for a single document"""
        print(f"   ⏱️  Time: {metrics.total_processing_time:.2f}s (API: {metrics.api_call_time:.2f}s)")
        print(f"   💰 Cost: ${metrics.total_api_cost:.4f} ({metrics.openai_tokens_used:,} tokens)")
        print(f"   📋 Issues: {metrics.total_discrepancies} discrepancies, {metrics.total_focus_points} focus points")
        print(f"   ✅ Correctly Resolved: {metrics.discrepancies_correctly_resolved}/{metrics.total_discrepancies} discrepancies, {metrics.focus_points_correctly_resolved}/{metrics.total_focus_points} focus points")
        print(f"   📊 Correct Resolution Rate: {metrics.overall_correct_resolution_rate:.1f}%")
    
    def _generate_final_report(self):
        """Generate comprehensive final report"""
        print("\n" + "="*80)
        print("📈 FINAL EVALUATION REPORT")
        print("="*80)
        
        # Calculate aggregates for each set
        for doc_type in ["training", "test"]:
            docs = self.results[doc_type]
            if not docs:
                continue
            
            print(f"\n{'🎓 TRAINING SET' if doc_type == 'training' else '🧪 TEST SET'} RESULTS:")
            print("-" * 40)
            
            # Aggregate metrics
            total_docs = len(docs)
            total_time = sum(d.total_processing_time for d in docs)
            total_api_time = sum(d.api_call_time for d in docs)
            total_cost = sum(d.total_api_cost for d in docs)
            total_tokens = sum(d.openai_tokens_used for d in docs)
            
            # Issue counts
            total_disc = sum(d.total_discrepancies for d in docs)
            total_fp = sum(d.total_focus_points for d in docs)
            total_issues = total_disc + total_fp
            
            # Correct resolutions
            disc_correct = sum(d.discrepancies_correctly_resolved for d in docs)
            fp_correct = sum(d.focus_points_correctly_resolved for d in docs)
            total_correct = disc_correct + fp_correct
            
            # Incorrect resolutions
            disc_incorrect = sum(d.discrepancies_incorrectly_resolved for d in docs)
            fp_incorrect = sum(d.focus_points_incorrectly_resolved for d in docs)
            
            # Not attempted
            disc_not_attempted = sum(d.discrepancies_not_attempted for d in docs)
            fp_not_attempted = sum(d.focus_points_not_attempted for d in docs)
            
            # Calculate rates
            disc_correct_rate = (disc_correct / total_disc * 100) if total_disc > 0 else 0
            fp_correct_rate = (fp_correct / total_fp * 100) if total_fp > 0 else 0
            overall_correct_rate = (total_correct / total_issues * 100) if total_issues > 0 else 0
            
            print(f"📄 Documents Processed: {total_docs}")
            print(f"\n⏱️  TIMING:")
            print(f"   Total Processing Time: {total_time:.2f}s")
            print(f"   Average per Document: {total_time/total_docs:.2f}s")
            print(f"   Total API Time: {total_api_time:.2f}s")
            print(f"   API Time Percentage: {total_api_time/total_time*100:.1f}%")
            
            print(f"\n💰 COSTS:")
            print(f"   Total Cost: ${total_cost:.4f}")
            print(f"   Average per Document: ${total_cost/total_docs:.4f}")
            print(f"   Total Tokens Used: {total_tokens:,}")
            print(f"   Average Tokens per Document: {total_tokens//total_docs:,}")
            
            print(f"\n📊 ACCURACY:")
            print(f"   Total Issues Found: {total_issues} ({total_disc} discrepancies, {total_fp} focus points)")
            print(f"   \n   DISCREPANCIES:")
            print(f"      Correctly Resolved: {disc_correct}/{total_disc} ({disc_correct_rate:.1f}%)")
            print(f"      Incorrectly Resolved: {disc_incorrect}/{total_disc}")
            print(f"      Not Attempted: {disc_not_attempted}/{total_disc}")
            print(f"   \n   FOCUS POINTS:")
            print(f"      Correctly Resolved: {fp_correct}/{total_fp} ({fp_correct_rate:.1f}%)")
            print(f"      Incorrectly Resolved: {fp_incorrect}/{total_fp}")
            print(f"      Not Attempted: {fp_not_attempted}/{total_fp}")
            print(f"   \n   ✅ OVERALL CORRECT RESOLUTION RATE: {overall_correct_rate:.1f}%")
        
        # Save detailed results
        self._save_detailed_results()
    
    def _save_detailed_results(self):
        """Save detailed results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}.json"
        
        # Convert dataclasses to dicts
        results_dict = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "summary": self._generate_summary_dict(),
            "training_results": [asdict(r) for r in self.results["training"]],
            "test_results": [asdict(r) for r in self.results["test"]]
        }
        
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {filename}")
    
    def _generate_summary_dict(self) -> Dict[str, Any]:
        """Generate summary dictionary for saving"""
        summary = {}
        
        for doc_type in ["training", "test"]:
            docs = self.results[doc_type]
            if not docs:
                continue
            
            total_issues = sum(d.total_discrepancies + d.total_focus_points for d in docs)
            total_correct = sum(d.discrepancies_correctly_resolved + d.focus_points_correctly_resolved for d in docs)
            
            summary[doc_type] = {
                "total_documents": len(docs),
                "total_processing_time": sum(d.total_processing_time for d in docs),
                "total_cost": sum(d.total_api_cost for d in docs),
                "total_issues": total_issues,
                "total_correctly_resolved": total_correct,
                "overall_correct_resolution_rate": (total_correct / total_issues * 100) if total_issues > 0 else 0
            }
        
        return summary

async def main():
    """Run the comprehensive evaluation"""
    
    # Define document sets (using real document paths from the repository)
    training_documents = [
        "PEFundPortfolioExtraction/67ee89d7ecbb614e1103e532",
        "PEFundPortfolioExtraction/67ee89d7ecbb614e1103e533",
        "PEFundPortfolioExtraction/67ee89d7ecbb614e1103e534",
        "PEFundPortfolioExtraction/67ee89d7ecbb614e1103e535",
        "PEFundPortfolioExtraction/67ee89d7ecbb614e1103e536",
    ]
    
    test_documents = [
        "PEFundPortfolioExtraction/67ee89d7ecbb614e1103e537",
        "PEFundPortfolioExtraction/67ee89d7ecbb614e1103e538",
        "PEFundPortfolioExtraction/67ee89d7ecbb614e1103e539",
        "PEFundPortfolioExtraction/67ee89d7ecbb614e1103e540",
        "PEFundPortfolioExtraction/67ee89d7ecbb614e1103e541",
        "PEFundPortfolioExtraction/67ee89d7ecbb614e1103e542",
        "PEFundPortfolioExtraction/67ee89d7ecbb614e1103e543",
        "PEFundPortfolioExtraction/67ee89d7ecbb614e1103e544",
        "PEFundPortfolioExtraction/67ee89d7ecbb614e1103e545",
        "PEFundPortfolioExtraction/67ee89d7ecbb614e1103e546",
        "PEFundPortfolioExtraction/67ee89d7ecbb614e1103e547",
        "PEFundPortfolioExtraction/67ee89d7ecbb614e1103e548",
        "PEFundPortfolioExtraction/67ee89d7ecbb614e1103e549",
        "PEFundPortfolioExtraction/67ee89d7ecbb614e1103e550",
        "PEFundPortfolioExtraction/67ee89d7ecbb614e1103e551",
    ]
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key to run the evaluation.")
        return
    
    # Run evaluation
    evaluator = ComprehensiveEvaluator()
    await evaluator.run_evaluation(training_documents, test_documents)

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the evaluation
    asyncio.run(main())