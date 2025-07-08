"""
Tetrix AI Feedback Loop System - Main Entry Point
Integrated with Analytics Microservice for Real-World Financial Document Processing
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tetrix_feedback_loop.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import our core components
from analytics_client import TetrixAnalyticsClient, create_analytics_client
from intelligent_feedback_loop import IntelligentFeedbackLoopSystem
from evaluation_system import EvaluationSystem

class TetrixProductionSystem:
    """Production system integrating Grant's analytics with AI feedback loop"""
    
    def __init__(self, use_mock_analytics: bool = False):
        self.use_mock_analytics = use_mock_analytics
        self.analytics_client = None
        self.feedback_loop = None
        self.evaluation_system = None
        self.processing_metrics = {
            "documents_processed": 0,
            "discrepancies_found": 0,
            "corrections_applied": 0,
            "total_processing_time": 0.0,
            "average_improvement_score": 0.0
        }
    
    async def initialize(self):
        """Initialize all system components"""
        logger.info("Initializing Tetrix AI Feedback Loop System...")
        
        # Initialize analytics client
        self.analytics_client = create_analytics_client(use_mock=self.use_mock_analytics)
        
        # Initialize intelligent feedback loop system with AI agent
        self.feedback_loop = IntelligentFeedbackLoopSystem(fast_mode=False)
        
        # Initialize evaluation system
        self.evaluation_system = EvaluationSystem()
        
        logger.info("System components initialized successfully")
    
    async def test_connectivity(self) -> Dict[str, Any]:
        """Test connectivity to Grant's analytics service"""
        logger.info("Testing connectivity to Grant's analytics service...")
        
        async with self.analytics_client:
            connectivity_result = await self.analytics_client.test_connection()
            
            if connectivity_result.get("connected", False):
                logger.info("Successfully connected to Grant's analytics service")
                return {
                    "success": True,
                    "connection_status": "connected",
                    "response_time": connectivity_result.get("response_time", 0),
                    "vpn_status": connectivity_result.get("vpn_status", "unknown")
                }
            else:
                logger.warning("Failed to connect to Grant's analytics service")
                return {
                    "success": False,
                    "connection_status": "failed",
                    "error": connectivity_result.get("error", "Unknown connection error"),
                    "vpn_check": connectivity_result.get("vpn_check", "Check network connectivity")
                }
    
    async def process_document(self, document_path: str, enable_ai_agent: bool = True) -> Dict[str, Any]:
        """Process a single document through the complete feedback loop"""
        
        start_time = time.time()
        
        logger.info(f"🚀 Processing document with AI Agent: {document_path}")
        
        try:
            if enable_ai_agent:
                # Use the new AI Agent-powered feedback loop system
                async with self.feedback_loop:
                    feedback_result = await self.feedback_loop.process_document_with_feedback_loop(document_path)
                
                # Update metrics
                self.processing_metrics["documents_processed"] += 1
                self.processing_metrics["discrepancies_found"] += feedback_result.original_issues
                self.processing_metrics["corrections_applied"] += len(feedback_result.corrections_applied)
                self.processing_metrics["total_processing_time"] += feedback_result.processing_time
                
                # Update average improvement score
                current_avg = self.processing_metrics["average_improvement_score"]
                docs_processed = self.processing_metrics["documents_processed"]
                self.processing_metrics["average_improvement_score"] = (
                    (current_avg * (docs_processed - 1) + feedback_result.improvement_percentage / 100) 
                    / docs_processed
                )
                
                # # Debug: log type and value of corrections_applied
                # logger.info(f"corrections_applied type: {type(feedback_result.corrections_applied)}, value: {feedback_result.corrections_applied}")
                # print(f"corrections_applied type: {type(feedback_result.corrections_applied)}, value: {feedback_result.corrections_applied}")
                # # Log before and after for each correction
                # for correction in feedback_result.corrections_applied:
                #     field = correction.get('field', 'unknown')
                #     original = correction.get('original_value', 'unknown')
                #     corrected = correction.get('corrected_value', 'unknown')
                #     log_msg = f"Correction applied to '{field}': BEFORE='{original}' AFTER='{corrected}'"
                #     logger.info(log_msg)
                #     print(log_msg)
                
                # Convert feedback loop result to our expected format
                return {
                    "success": feedback_result.validation_successful,
                    "document_path": document_path,
                    "total_issues": feedback_result.original_issues,
                    "discrepancies": feedback_result.original_issues,  # Combined for backward compatibility
                    "focus_points": 0,
                    "corrections_applied": len(feedback_result.corrections_applied),
                    "corrections": feedback_result.corrections_applied,
                    "improvement_score": feedback_result.improvement_percentage / 100,
                    "processing_time": feedback_result.processing_time,
                    "processing_mode": "ai_agent_powered",
                    "agent_reasoning": feedback_result.agent_reasoning,
                    "revalidation_results": {
                        "original_issues": feedback_result.original_issues,
                        "remaining_issues": feedback_result.corrected_issues,
                        "issues_resolved": feedback_result.original_issues - feedback_result.corrected_issues,
                        "actual_improvement_percentage": feedback_result.improvement_percentage,
                        "validation_successful": feedback_result.validation_successful
                    },
                    "next_actions": feedback_result.next_actions,
                    "sample_issues": []  # Would be populated from original analysis
                }
            
            # else:
            #     # Fallback: Original rule-based approach
            #     async with self.analytics_client:
            #         analytics_response = await self.analytics_client.get_discrepancies_for_document(document_path)
                
            #     total_issues = len(analytics_response.discrepancies) + len(analytics_response.focus_points)
                
            #     if total_issues == 0:
            #         logger.info(f"No issues found in document {document_path}")
            #         return {
            #             "success": True,
            #             "document_path": document_path,
            #             "total_issues": 0,
            #             "corrections_applied": 0,
            #             "improvement_score": 1.0,
            #             "processing_time": time.time() - start_time,
            #             "processing_mode": "rule_based_fallback",
            #             "message": "Document is clean - no discrepancies or focus points detected"
            #         }
                
            #     logger.info(f"Found {len(analytics_response.discrepancies)} discrepancies and {len(analytics_response.focus_points)} focus points")
                
            #     # Simple rule-based processing for fallback
            #     corrections_applied = min(total_issues // 2, 10)  # Fix up to half the issues, max 10
            #     improvement_score = corrections_applied / total_issues if total_issues > 0 else 0
                
            #     return {
            #         "success": True,
            #         "document_path": document_path,
            #         "total_issues": total_issues,
            #         "discrepancies": len(analytics_response.discrepancies),
            #         "focus_points": len(analytics_response.focus_points),
            #         "corrections_applied": corrections_applied,
            #         "improvement_score": improvement_score,
            #         "processing_time": time.time() - start_time,
            #         "processing_mode": "rule_based_fallback",
            #         "revalidation_results": {
            #             "original_issues": total_issues,
            #             "remaining_issues": total_issues - corrections_applied,
            #             "issues_resolved": corrections_applied,
            #             "actual_improvement_percentage": improvement_score * 100,
            #             "validation_successful": True
            #         }
            #     }
            
            
        except Exception as e:
            logger.error(f"Error processing document {document_path}: {e}")
            return {
                "success": False,
                "document_path": document_path,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def batch_process_documents(self, document_paths: List[str], enable_ai_agent: bool = True) -> Dict[str, Any]:
        """Process multiple documents and provide comprehensive analysis"""
        
        logger.info(f"Starting batch processing of {len(document_paths)} documents")
        
        batch_start_time = time.time()
        results = []
        
        for doc_path in document_paths:
            result = await self.process_document(doc_path, enable_ai_agent)
            results.append(result)
            
            # Log progress
            if result["success"]:
                logger.info(f"SUCCESS {doc_path}: {result['total_issues']} issues, {result['improvement_score']:.1%} improvement")
            else:
                logger.error(f"FAILED {doc_path}: {result.get('error', 'Unknown error')}")
        
        batch_processing_time = time.time() - batch_start_time
        
        # Analyze batch results
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]
        
        if successful_results:
            total_issues_found = sum(r["total_issues"] for r in successful_results)
            total_corrections = sum(r["corrections_applied"] for r in successful_results)
            avg_improvement = sum(r["improvement_score"] for r in successful_results) / len(successful_results)
            avg_processing_time = sum(r["processing_time"] for r in successful_results) / len(successful_results)
        else:
            total_issues_found = total_corrections = avg_improvement = avg_processing_time = 0
        
        return {
            "batch_summary": {
                "total_documents": len(document_paths),
                "successful_documents": len(successful_results),
                "failed_documents": len(failed_results),
                "total_issues_found": total_issues_found,
                "total_corrections_applied": total_corrections,
                "average_improvement_score": avg_improvement,
                "average_processing_time": avg_processing_time,
                "batch_processing_time": batch_processing_time
            },
            "individual_results": results,
            "system_metrics": self.processing_metrics
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "system_info": {
                "use_mock_analytics": self.use_mock_analytics,
                "llm_api_available": bool(os.getenv("OPENAI_API_KEY")),
                "timestamp": datetime.now().isoformat()
            },
            "processing_metrics": self.processing_metrics,
            "components": {
                "analytics_client": "initialized" if self.analytics_client else "not_initialized",
                "feedback_loop": "initialized" if self.feedback_loop else "not_initialized",
                "evaluation_system": "initialized" if self.evaluation_system else "not_initialized"
            }
        }

async def run_sample_integration_test():
    """Run a sample integration test with real documents"""
    
    print("TETRIX AI FEEDBACK LOOP SYSTEM")
    print("=" * 80)
    print("Integrated Real-World Testing with Grant's Analytics Microservice")
    print("Priority: Implementing feedback loop to fix discrepancies before consolidation")
    print("=" * 80)
    
    # Initialize system
    print("\nINITIALIZING SYSTEM...")
    
    has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
    use_mock = False
    
    system = TetrixProductionSystem(use_mock_analytics=use_mock)
    await system.initialize()
    
    analytics_mode = "Mock" if use_mock else "Real Grant's Service"
    print(f"   Analytics Mode: {analytics_mode}")
    print(f"   LLM Integration: {'Enabled' if has_openai_key else 'Disabled (Mock Mode)'}")
    
    # Test connectivity
    print("\nTESTING CONNECTIVITY...")
    connectivity = await system.test_connectivity()
    
    if connectivity["success"]:
        print(f"   Connected to analytics service")
        print(f"   Response Time: {connectivity.get('response_time', 0):.2f}s")
    else:
        print(f"   Connection issue: {connectivity.get('error', 'Unknown')}")
        if not use_mock:
            print("   Suggestion: Check VPN connection to Grant's internal network")
    
    # Test with real document paths
    print("\nPROCESSING REAL FINANCIAL DOCUMENTS...")
    
    test_documents = [
        "PEFundPortfolioExtraction/67ee89d7ecbb614e1103e533",  # ABRY Partners VIII
        # "PEFundPortfolioExtraction/67ee89d7ecbb614e1103e534",
        # "PEFundPortfolioExtraction/67ee89d7ecbb614e1103e535"
    ]
    
    # Process documents with AI Agent
    batch_result = await system.batch_process_documents(
        test_documents, 
        enable_ai_agent=True  # Always use AI agent - it has fallbacks
    )
    
    # Display results
    summary = batch_result["batch_summary"]
    print(f"\nBATCH PROCESSING RESULTS:")
    print(f"   Documents Processed: {summary['successful_documents']}/{summary['total_documents']}")
    print(f"   Total Issues Found: {summary['total_issues_found']}")
    print(f"   Corrections Applied: {summary['total_corrections_applied']}")
    print(f"   ESTIMATED Improvement: {summary['average_improvement_score']:.1%} (conservative estimate)")
    print(f"   Average Processing Time: {summary['average_processing_time']:.2f}s")
    print(f"   Batch Total Time: {summary['batch_processing_time']:.2f}s")
    print(f"   NOTE: Cannot measure actual improvement without re-analyzing corrected documents")
    
    # Show individual document results
    print(f"\nINDIVIDUAL DOCUMENT RESULTS:")
    for result in batch_result["individual_results"]:
        if result["success"]:
            print(f"   SUCCESS {result['document_path']}:")
            print(f"      Issues: {result['total_issues']} ({result['discrepancies']} discrepancies, {result['focus_points']} focus points)")
            print(f"      Corrections: {result['corrections_applied']}")
            print(f"      ESTIMATED Improvement: {result['improvement_score']:.1%} (conservative estimate)")
            print(f"      Time: {result['processing_time']:.2f}s")
            
            # Show re-validation results if available
            if result.get("revalidation_results"):
                revalidation = result["revalidation_results"]
                print(f"      Re-validation: {revalidation['original_issues']} → {revalidation['remaining_issues']} issues "
                      f"({revalidation['issues_resolved']} estimated resolved)")
                print(f"      NOTE: {revalidation['actual_improvement_percentage']:.1f}% is an ESTIMATE, not actual measurement")
            elif result.get("estimated_improvement_score"):
                print(f"      Estimated Improvement: {result['estimated_improvement_score']:.1%} (not validated)")
            
            # Show sample issues
            if result.get("sample_issues"):
                print(f"      Sample Issues:")
                for issue in result["sample_issues"]:
                    print(f"        • {issue['field']}: {issue['issue']} (confidence: {issue['confidence']:.1%})")
            
            # Show corrections applied (improved document preview)
            if result.get("corrections_applied", 0) > 0 and result.get("processing_mode") in ["llm_powered", "rule_based"]:
                print(f"      Corrections Applied:")
                # Save improved document to file for inspection
                doc_id = result['document_path'].split('/')[-1]
                improved_doc_file = f"improved_document_{doc_id}.json"
                
                # Create improved document data to save
                improved_doc_data = {
                    "original_document_path": result['document_path'],
                    "processing_timestamp": datetime.now().isoformat(),
                    "original_issues": result['total_issues'],
                    "corrections_applied": result['corrections_applied'],
                    "improvement_score": result['improvement_score'],
                    "processing_mode": result.get('processing_mode', 'unknown'),
                    "corrections": [],
                    "revalidation_results": result.get('revalidation_results')
                }
                
                # Add actual correction details from LLM processing
                if 'corrections' in result and result['corrections']:
                    # Use actual corrections from LLM processing
                    for correction in result['corrections']:
                        correction_entry = {
                            "field": correction['field'],
                            "original_value": correction.get('original_value', 'unknown'),
                            "corrected_value": correction.get('corrected_value', 'corrected'),
                            "confidence": correction.get('confidence', 0.95),
                            "reasoning": correction.get('reasoning', 'LLM-powered correction'),
                            "corrected": True,
                            "correction_method": result.get('processing_mode', 'unknown')
                        }
                        improved_doc_data["corrections"].append(correction_entry)
                        print(f"        • {correction['field']}: {correction.get('original_value', 'unknown')} → {correction.get('corrected_value', 'corrected')}")
                elif 'sample_issues' in result:
                    # Fallback to sample issues for rule-based processing
                    for i, issue in enumerate(result['sample_issues'][:3]):  # Show first 3 corrections
                        correction = {
                            "field": issue['field'],
                            "original_issue": issue['issue'],
                            "confidence": issue['confidence'],
                            "corrected": True,
                            "correction_method": result.get('processing_mode', 'unknown')
                        }
                        improved_doc_data["corrections"].append(correction)
                
                try:
                    import json
                    with open(improved_doc_file, 'w') as f:
                        json.dump(improved_doc_data, f, indent=2)
                    print(f"      Improved document saved to: {improved_doc_file}")
                except Exception as e:
                    print(f"      Could not save improved document: {e}")
                    
        else:
            print(f"   FAILED {result['document_path']}: {result.get('error', 'Unknown error')}")
    
    # System status
    print(f"\nSYSTEM STATUS:")
    status = system.get_system_status()
    metrics = status["processing_metrics"]
    
    print(f"   Total Documents Processed: {metrics['documents_processed']}")
    print(f"   Total Discrepancies Found: {metrics['discrepancies_found']}")
    print(f"   Total Corrections Applied: {metrics['corrections_applied']}")
    print(f"   Average Improvement Score: {metrics['average_improvement_score']:.1%}")
    

if __name__ == "__main__":
    print("Starting Tetrix AI Feedback Loop System Integration Test...")
    asyncio.run(run_sample_integration_test())