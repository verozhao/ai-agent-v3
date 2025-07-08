"""
Tetrix AI Feedback Loop System
Complete integration between analytics microservice and AI-powered document correction
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

from analytics_client import TetrixAnalyticsClient, create_analytics_client, AnalyticsResponse
from discrepancy_processor import CombinedIssueProcessor, ProcessingResult
from financial_agent import FinancialDocumentSpecialistAgent, FinancialLLMEngine

logger = logging.getLogger(__name__)

@dataclass
class FeedbackLoopResult:
    """Complete result from feedback loop processing"""
    document_id: str
    original_document: Dict[str, Any]
    improved_document: Dict[str, Any]
    analytics_before: AnalyticsResponse
    analytics_after: Optional[AnalyticsResponse]
    processing_results: Dict[str, Any]
    improvement_metrics: Dict[str, Any]
    processing_time: float
    feedback_loop_successful: bool
    error_message: Optional[str] = None

class TetrixFeedbackLoopSystem:
    """
    Main feedback loop system integrating Grant's analytics with AI correction
    
    This system implements the complete workflow described in the meeting:
    1. Extract document analysis from Grant's analytics API
    2. Process discrepancies and focus points with specialized AI
    3. Generate improved document
    4. Measure improvement
    5. Provide learning feedback
    """
    
    def __init__(self, use_mock_analytics: bool = None):
        # Initialize components
        self.analytics_client = create_analytics_client(use_mock_analytics)
        
        # Initialize LLM components only if API keys are available
        import os
        has_llm_key = bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
        
        if has_llm_key:
            self.llm_engine = FinancialLLMEngine()
            self.issue_processor = CombinedIssueProcessor(self.llm_engine)
            self.financial_agent = FinancialDocumentSpecialistAgent()
            logger.info("LLM-powered processing enabled")
        else:
            self.llm_engine = None
            self.issue_processor = None
            self.financial_agent = None
            logger.info("Rule-based processing mode (no LLM API keys found)")
        
        # System metrics
        self.system_metrics = {
            "documents_processed": 0,
            "total_issues_found": 0,
            "total_corrections_applied": 0,
            "avg_improvement_score": 0.0,
            "feedback_loops_completed": 0,
            "system_uptime": datetime.now().isoformat()
        }
        
        logger.info("Tetrix Feedback Loop System initialized")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.analytics_client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.analytics_client.__aexit__(exc_type, exc_val, exc_tb)
    
    async def test_system_connectivity(self) -> Dict[str, Any]:
        """Test all system components"""
        connectivity_results = {
            "timestamp": datetime.now().isoformat(),
            "analytics_service": {},
            "llm_engine": {},
            "overall_status": "unknown"
        }
        
        # Test analytics service
        try:
            analytics_test = await self.analytics_client.test_connection()
            connectivity_results["analytics_service"] = analytics_test
        except Exception as e:
            connectivity_results["analytics_service"] = {
                "connected": False,
                "error": str(e)
            }
        
        # Test LLM engine
        if self.llm_engine:
            try:
                test_response = await self.llm_engine.reason_about_document(
                    {"test": "connectivity_test"}, "test"
                )
                connectivity_results["llm_engine"] = {
                    "available": True,
                    "model_used": test_response.get("model_used", "unknown"),
                    "response_time": test_response.get("response_time", 0)
                }
            except Exception as e:
                connectivity_results["llm_engine"] = {
                    "available": False,
                    "error": str(e)
                }
        else:
            connectivity_results["llm_engine"] = {
                "available": False,
                "error": "LLM engine not initialized (no API keys found)"
            }
        
        # Determine overall status
        analytics_ok = connectivity_results["analytics_service"].get("connected", False)
        llm_ok = connectivity_results["llm_engine"].get("available", False)
        
        if analytics_ok and llm_ok:
            connectivity_results["overall_status"] = "ready"
        elif analytics_ok or llm_ok:
            connectivity_results["overall_status"] = "partial"
        else:
            connectivity_results["overall_status"] = "failed"
        
        return connectivity_results
    
    async def run_feedback_loop(self, extracted_document: Dict[str, Any], 
                               document_path: str, 
                               client_entity_or_org: str = "client_entity",
                               ce_or_org_id: str = "default",
                               re_analyze_after_correction: bool = True) -> FeedbackLoopResult:
        """
        Run the complete feedback loop for document improvement
        
        This is the main entry point that implements the full workflow
        """
        start_time = time.time()
        document_id = f"doc_{int(time.time())}_{hash(str(extracted_document)) % 10000}"
        
        logger.info(f"Starting feedback loop for document {document_id} at path {document_path}")
        
        try:
            # Step 1: Get initial analytics (discrepancies and focus points)
            logger.info("Step 1: Analyzing document with Grant's analytics system...")
            analytics_before = await self.analytics_client.get_discrepancies_for_document(
                doc_path=document_path,
                client_entity_or_org=client_entity_or_org,
                ce_or_org_id=ce_or_org_id
            )
            
            initial_issues = len(analytics_before.discrepancies) + len(analytics_before.focus_points)
            logger.info(f"Found {len(analytics_before.discrepancies)} discrepancies and "
                       f"{len(analytics_before.focus_points)} focus points")
            
            # Step 2: Process issues with specialized AI
            logger.info("Step 2: Processing issues with specialized AI agents...")
            processing_results = await self.issue_processor.process_analytics_response(
                analytics_before, extracted_document
            )
            
            improved_document = processing_results["corrected_document"]
            corrections_applied = processing_results["summary"]["corrections_applied"]
            
            logger.info(f"Applied {corrections_applied} corrections to document")
            
            # Step 3: Re-analyze improved document (optional but recommended)
            analytics_after = None
            if re_analyze_after_correction and corrections_applied > 0:
                logger.info("Step 3: Re-analyzing improved document...")
                
                # In a real system, we'd push the improved document and get new analytics
                # For now, we simulate this or call analytics again
                analytics_after = await self._analyze_improved_document(
                    improved_document, document_path, client_entity_or_org, ce_or_org_id
                )
                
                final_issues = len(analytics_after.discrepancies) + len(analytics_after.focus_points)
                logger.info(f"After correction: {final_issues} issues remaining")
            
            # Step 4: Calculate improvement metrics
            improvement_metrics = self._calculate_improvement_metrics(
                analytics_before, analytics_after, processing_results
            )
            
            # Step 5: Update system metrics
            self._update_system_metrics(initial_issues, corrections_applied, improvement_metrics)
            
            processing_time = time.time() - start_time
            
            result = FeedbackLoopResult(
                document_id=document_id,
                original_document=extracted_document,
                improved_document=improved_document,
                analytics_before=analytics_before,
                analytics_after=analytics_after,
                processing_results=processing_results,
                improvement_metrics=improvement_metrics,
                processing_time=processing_time,
                feedback_loop_successful=True
            )
            
            logger.info(f"Feedback loop completed successfully for {document_id} "
                       f"in {processing_time:.2f}s with {corrections_applied} corrections")
            
            return result
            
        except Exception as e:
            logger.error(f"Feedback loop failed for {document_id}: {e}")
            
            return FeedbackLoopResult(
                document_id=document_id,
                original_document=extracted_document,
                improved_document=extracted_document,  # No improvement on error
                analytics_before=AnalyticsResponse(
                    document_path=document_path,
                    document_type="unknown",
                    discrepancies=[],
                    focus_points=[],
                    consolidation_metadata={},
                    historical_data={},
                    response_timestamp=datetime.now(),
                    processing_time_ms=0
                ),
                analytics_after=None,
                processing_results={},
                improvement_metrics={},
                processing_time=time.time() - start_time,
                feedback_loop_successful=False,
                error_message=str(e)
            )
    
    async def _analyze_improved_document(self, improved_document: Dict[str, Any],
                                       document_path: str, client_entity_or_org: str,
                                       ce_or_org_id: str) -> AnalyticsResponse:
        """
        Analyze the improved document to measure effectiveness
        In a real system, this would push the document and get fresh analytics
        """
        
        try:
            return await self.analytics_client.get_discrepancies_for_document(
                doc_path=document_path,
                client_entity_or_org=client_entity_or_org,
                ce_or_org_id=ce_or_org_id
            )
        except Exception as e:
            logger.warning(f"Could not re-analyze improved document: {e}")
            
            # Return empty analytics on error
            return AnalyticsResponse(
                document_path=document_path,
                document_type="unknown",
                discrepancies=[],
                focus_points=[],
                consolidation_metadata={"re_analysis": "failed"},
                historical_data={},
                response_timestamp=datetime.now(),
                processing_time_ms=0
            )
    
    def _calculate_improvement_metrics(self, analytics_before: AnalyticsResponse,
                                     analytics_after: Optional[AnalyticsResponse],
                                     processing_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics showing improvement from feedback loop"""
        
        initial_discrepancies = len(analytics_before.discrepancies)
        initial_focus_points = len(analytics_before.focus_points)
        initial_total = initial_discrepancies + initial_focus_points
        
        corrections_applied = processing_results.get("summary", {}).get("corrections_applied", 0)
        issues_flagged = processing_results.get("summary", {}).get("issues_flagged", 0)
        
        metrics = {
            "initial_issues": {
                "discrepancies": initial_discrepancies,
                "focus_points": initial_focus_points,
                "total": initial_total
            },
            "corrections_applied": corrections_applied,
            "issues_flagged": issues_flagged,
            "correction_rate": corrections_applied / max(initial_total, 1),
            "processing_effective": corrections_applied > 0
        }
        
        # If we have after-analysis, calculate reduction
        if analytics_after:
            final_discrepancies = len(analytics_after.discrepancies)
            final_focus_points = len(analytics_after.focus_points)
            final_total = final_discrepancies + final_focus_points
            
            metrics["final_issues"] = {
                "discrepancies": final_discrepancies,
                "focus_points": final_focus_points,
                "total": final_total
            }
            
            metrics["improvement"] = {
                "issues_resolved": initial_total - final_total,
                "resolution_rate": (initial_total - final_total) / max(initial_total, 1),
                "discrepancies_resolved": initial_discrepancies - final_discrepancies,
                "focus_points_resolved": initial_focus_points - final_focus_points
            }
            
            # Overall improvement score (0-1)
            improvement_score = max(0, (initial_total - final_total) / max(initial_total, 1))
            metrics["improvement_score"] = improvement_score
        else:
            # Estimate improvement based on corrections applied
            estimated_improvement = min(corrections_applied / max(initial_total, 1), 1.0)
            metrics["estimated_improvement_score"] = estimated_improvement
        
        return metrics
    
    def _update_system_metrics(self, issues_found: int, corrections_applied: int, 
                             improvement_metrics: Dict[str, Any]):
        """Update system-wide metrics"""
        self.system_metrics["documents_processed"] += 1
        self.system_metrics["total_issues_found"] += issues_found
        self.system_metrics["total_corrections_applied"] += corrections_applied
        self.system_metrics["feedback_loops_completed"] += 1
        
        # Update average improvement score
        improvement_score = improvement_metrics.get("improvement_score") or improvement_metrics.get("estimated_improvement_score", 0)
        current_avg = self.system_metrics["avg_improvement_score"]
        docs_processed = self.system_metrics["documents_processed"]
        
        self.system_metrics["avg_improvement_score"] = (
            (current_avg * (docs_processed - 1) + improvement_score) / docs_processed
        )
    
    async def batch_process_documents(self, documents: List[Dict[str, Any]], 
                                    document_paths: List[str],
                                    client_entity_or_org: str = "client_entity",
                                    ce_or_org_id: str = "default") -> List[FeedbackLoopResult]:
        """Process multiple documents through the feedback loop"""
        
        if len(documents) != len(document_paths):
            raise ValueError("Number of documents must match number of document paths")
        
        logger.info(f"Starting batch processing of {len(documents)} documents")
        
        results = []
        
        for i, (document, doc_path) in enumerate(zip(documents, document_paths)):
            logger.info(f"Processing document {i+1}/{len(documents)}: {doc_path}")
            
            try:
                result = await self.run_feedback_loop(
                    document, doc_path, client_entity_or_org, ce_or_org_id
                )
                results.append(result)
                
                # Brief pause between documents to avoid overwhelming the system
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Failed to process document {doc_path}: {e}")
                # Continue with other documents
                continue
        
        logger.info(f"Batch processing completed: {len(results)}/{len(documents)} successful")
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "system_metrics": self.system_metrics,
            "analytics_client_metrics": self.analytics_client.get_metrics(),
            "status_timestamp": datetime.now().isoformat()
        }
        
        # Add issue processor stats if available
        if self.issue_processor:
            status["issue_processor_stats"] = self.issue_processor.get_processing_stats()
        else:
            status["issue_processor_stats"] = "not_initialized"
            
        # Add financial agent status if available
        if self.financial_agent:
            try:
                status["financial_agent_status"] = self.financial_agent.get_agent_status()
            except AttributeError:
                status["financial_agent_status"] = "available_but_no_status_method"
        else:
            status["financial_agent_status"] = "not_initialized"
            
        return status
    
    async def evaluate_system_performance(self, test_documents: List[Dict[str, Any]], 
                                        ground_truths: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate system performance against ground truth data
        This is crucial for measuring the effectiveness of the feedback loop
        """
        
        if len(test_documents) != len(ground_truths):
            raise ValueError("Number of test documents must match number of ground truths")
        
        evaluation_results = {
            "total_documents": len(test_documents),
            "successful_processing": 0,
            "accuracy_scores": [],
            "improvement_scores": [],
            "processing_times": [],
            "detailed_results": []
        }
        
        for i, (document, ground_truth) in enumerate(zip(test_documents, ground_truths)):
            doc_path = f"test_doc_{i}"
            
            try:
                # Run feedback loop
                result = await self.run_feedback_loop(document, doc_path)
                
                if result.feedback_loop_successful:
                    evaluation_results["successful_processing"] += 1
                    
                    # Calculate accuracy against ground truth
                    accuracy = self._calculate_document_accuracy(
                        result.improved_document, ground_truth
                    )
                    evaluation_results["accuracy_scores"].append(accuracy)
                    
                    # Get improvement score
                    improvement_score = result.improvement_metrics.get("improvement_score", 0)
                    evaluation_results["improvement_scores"].append(improvement_score)
                    
                    evaluation_results["processing_times"].append(result.processing_time)
                    
                    evaluation_results["detailed_results"].append({
                        "document_id": result.document_id,
                        "accuracy": accuracy,
                        "improvement_score": improvement_score,
                        "corrections_applied": result.processing_results.get("summary", {}).get("corrections_applied", 0),
                        "processing_time": result.processing_time
                    })
                
            except Exception as e:
                logger.error(f"Evaluation failed for document {i}: {e}")
                continue
        
        # Calculate summary statistics
        if evaluation_results["accuracy_scores"]:
            evaluation_results["avg_accuracy"] = sum(evaluation_results["accuracy_scores"]) / len(evaluation_results["accuracy_scores"])
            evaluation_results["avg_improvement"] = sum(evaluation_results["improvement_scores"]) / len(evaluation_results["improvement_scores"])
            evaluation_results["avg_processing_time"] = sum(evaluation_results["processing_times"]) / len(evaluation_results["processing_times"])
        else:
            evaluation_results["avg_accuracy"] = 0.0
            evaluation_results["avg_improvement"] = 0.0
            evaluation_results["avg_processing_time"] = 0.0
        
        evaluation_results["success_rate"] = evaluation_results["successful_processing"] / evaluation_results["total_documents"]
        
        return evaluation_results
    
    def _calculate_document_accuracy(self, improved_document: Dict[str, Any], 
                                   ground_truth: Dict[str, Any]) -> float:
        """Calculate accuracy of improved document against ground truth"""
        
        if not ground_truth:
            return 1.0
        
        correct_fields = 0
        total_fields = len(ground_truth)
        
        for field, expected_value in ground_truth.items():
            actual_value = improved_document.get(field)
            
            # Handle different value types
            if isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
                # Numeric comparison with tolerance
                tolerance = max(abs(expected_value) * 0.01, 1.0)  # 1% tolerance or minimum 1
                if abs(actual_value - expected_value) <= tolerance:
                    correct_fields += 1
            elif str(actual_value) == str(expected_value):
                correct_fields += 1
        
        return correct_fields / total_fields if total_fields > 0 else 1.0

# Export
__all__ = ["TetrixFeedbackLoopSystem", "FeedbackLoopResult"]