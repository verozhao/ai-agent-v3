"""
Comprehensive Test Suite for Tetrix AI Feedback Loop System
Real-world validation with Grant's analytics endpoints and document processing
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import os
import sys

from feedback_loop_system import TetrixFeedbackLoopSystem, FeedbackLoopResult
from analytics_client import create_analytics_client
from financial_agent import FinancialDocumentSpecialistAgent

logger = logging.getLogger(__name__)

class TestDataGenerator:
    """Generate realistic test documents based on meeting examples"""
    
    @staticmethod
    def get_abry_partners_test_documents() -> List[Dict[str, Any]]:
        """
        Generate test documents based on Abry Partners example from meeting
        Simulates the real document extraction scenarios
        """
        return [
            {
                "name": "Abry Partners V - Q4 2023 Fund Report",
                "document_path": "PEFundPortfolioExtraction/67ee89d7ecbb614e1103e533",
                "document_type": "fund_report",
                "client_entity": "client_entity",
                "ce_or_org_id": "abry_partners_v",
                "extracted_document": {
                    "fund_name": "Abry Partners V",
                    "reporting_date": "2023-12-31",
                    "total_fund_nav": 1250000000,  # May have discrepancies
                    "craneware_realized_value": 150000,
                    "craneware_unrealized_value": 0,
                    "fastmed_realized_value": None,  # Missing values mentioned in meeting
                    "fastmed_unrealized_value": 45000000,
                    "null_platform_technologies_value": 150000, 
                    "total_realized_gains": 125000000,
                    "total_unrealized_gains": 875000000,
                    "number_of_portfolio_companies": 45,
                    "vintage_year": 2020,
                    "fund_size": 2500000000
                },
                "ground_truth": {
                    "fund_name": "Abry Partners V",
                    "reporting_date": "2023-12-31", 
                    "total_fund_nav": 1250000000,
                    "craneware_realized_value": 650000,  # Corrected value
                    "craneware_unrealized_value": 0,
                    "fastmed_realized_value": 25000000,  # Should not be None
                    "fastmed_unrealized_value": 45000000,
                    "null_platform_technologies_value": 1150000,
                    "total_realized_gains": 125000000,
                    "total_unrealized_gains": 875000000,
                    "number_of_portfolio_companies": 45,
                    "vintage_year": 2020,
                    "fund_size": 2500000000
                },
                "expected_discrepancies": ["craneware_realized_value", "null_platform_technologies_value"],
                "expected_focus_points": ["fastmed_realized_value"]
            },
            {
                "name": "Abry Partners V - Q3 2023 Fund Report", 
                "document_path": "PEFundPortfolioExtraction/67ee89d7ecbb614e1103e534",
                "document_type": "fund_report",
                "client_entity": "client_entity",
                "ce_or_org_id": "abry_partners_v",
                "extracted_document": {
                    "fund_name": "Abry Partners V",
                    "reporting_date": "2023-09-30",
                    "total_fund_nav": 1180000000,
                    "craneware_realized_value": 650000,  # Consistent with previous correct value
                    "craneware_unrealized_value": 15000000,
                    "fastmed_realized_value": 25000000,
                    "fastmed_unrealized_value": 42000000,
                    "null_platform_technologies_value": 1150000,
                    "total_realized_gains": 115000000,
                    "total_unrealized_gains": 825000000,
                    "number_of_portfolio_companies": 44,
                    "vintage_year": 2020,
                    "fund_size": 2500000000
                },
                "ground_truth": {
                    "fund_name": "Abry Partners V",
                    "reporting_date": "2023-09-30",
                    "total_fund_nav": 1180000000,
                    "craneware_realized_value": 650000,
                    "craneware_unrealized_value": 15000000,
                    "fastmed_realized_value": 25000000,
                    "fastmed_unrealized_value": 42000000,
                    "null_platform_technologies_value": 1150000,
                    "total_realized_gains": 115000000,
                    "total_unrealized_gains": 825000000,
                    "number_of_portfolio_companies": 44,
                    "vintage_year": 2020,
                    "fund_size": 2500000000
                },
                "expected_discrepancies": [],  # This document should be clean
                "expected_focus_points": []
            },
            {
                "name": "Solamere Fund II - Capital Call Document",
                "document_path": "PEFundPortfolioExtraction/67ee89d7ecbb614e1103e535", 
                "document_type": "capital_call",
                "client_entity": "client_entity",
                "ce_or_org_id": "solamere_fund_ii",
                "extracted_document": {
                    "fund_name": "2024-01-15",  # Date in fund name field - classic field swap
                    "call_date": "Solamere Fund II",  # Fund name in date field
                    "call_amount": "fifty million",  # Text amount
                    "due_date": "2024-02-15",
                    "purpose": "New investment opportunity",
                    "total_fund_commitments": 750000000,
                    "total_called_to_date": 425000000,
                    "remaining_commitments": 275000000  # Should be 325M (750-425)
                },
                "ground_truth": {
                    "fund_name": "Solamere Fund II",
                    "call_date": "2024-01-15",
                    "call_amount": 50000000,
                    "due_date": "2024-02-15",
                    "purpose": "New investment opportunity",
                    "total_fund_commitments": 750000000,
                    "total_called_to_date": 425000000,
                    "remaining_commitments": 325000000
                },
                "expected_discrepancies": ["fund_name", "call_date", "remaining_commitments"],
                "expected_focus_points": ["call_amount"]
            }
        ]
    
    @staticmethod
    def get_complex_error_documents() -> List[Dict[str, Any]]:
        """Generate documents with multiple complex errors for stress testing"""
        return [
            {
                "name": "Multi-Error Vista Fund Document",
                "document_path": "PEFundPortfolioExtraction/complex_error_test_1",
                "document_type": "fund_report",
                "client_entity": "client_entity", 
                "ce_or_org_id": "vista_fund_viii",
                "extracted_document": {
                    "fund_name": "2020-01-15",  # Date in fund field
                    "fund_vintage": "Vista Equity Partners Fund VIII",  # Fund name in vintage
                    "first_close_date": "2020-01-15",
                    "final_close_date": "2019-12-31",  # Final before first (chronological error)
                    "fund_size": "nine billion six hundred million",  # Text amount
                    "q1_investments": 8,
                    "q2_investments": 12,
                    "q3_investments": 15,
                    "q4_investments": 18,
                    "total_investments": 50,  # Should be 53 (calculation error)
                    "portfolio_valuation": 8500000000,
                    "realized_proceeds": 1200000000,
                    "unrealized_value": 6800000000,  # Should be 7300M (valuation - realized)
                    "irr": "45%",  # Seems too high, likely error
                    "multiple": "2.1x"
                },
                "ground_truth": {
                    "fund_name": "Vista Equity Partners Fund VIII",
                    "fund_vintage": "2020",
                    "first_close_date": "2020-01-15",
                    "final_close_date": "2020-12-31",
                    "fund_size": 9600000000,
                    "q1_investments": 8,
                    "q2_investments": 12,
                    "q3_investments": 15,
                    "q4_investments": 18,
                    "total_investments": 53,
                    "portfolio_valuation": 8500000000,
                    "realized_proceeds": 1200000000,
                    "unrealized_value": 7300000000,
                    "irr": "18.5%",
                    "multiple": "2.1x"
                },
                "expected_discrepancies": ["fund_name", "fund_vintage", "final_close_date", "total_investments", "unrealized_value"],
                "expected_focus_points": ["fund_size", "irr"]
            }
        ]

class TetrixTestSuite:
    """Comprehensive test suite for the Tetrix AI feedback loop system"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.test_data_generator = TestDataGenerator()
        
        # Configure logging for tests
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    async def run_all_tests(self, use_mock_analytics: bool = None) -> Dict[str, Any]:
        """Run the complete test suite"""
        
        self.start_time = time.time()
        print("🚀 TETRIX AI FEEDBACK LOOP SYSTEM - COMPREHENSIVE TEST SUITE")
        print("=" * 80)
        print("Testing real-world integration with Grant's analytics microservice")
        print("=" * 80)
        
        # Test categories
        test_categories = [
            ("connectivity", self.test_system_connectivity),
            ("abry_partners_documents", self.test_abry_partners_documents),
            ("complex_error_handling", self.test_complex_error_handling),
            ("batch_processing", self.test_batch_processing),
            ("performance_metrics", self.test_performance_metrics),
            ("end_to_end_workflow", self.test_end_to_end_workflow)
        ]
        
        overall_results = {
            "test_suite_start": datetime.now().isoformat(),
            "test_categories": {},
            "summary": {},
            "system_info": {
                "mock_analytics": use_mock_analytics,
                "environment": self._get_environment_info()
            }
        }
        
        successful_categories = 0
        total_categories = len(test_categories)
        
        async with TetrixFeedbackLoopSystem(use_mock_analytics) as feedback_system:
            for category_name, test_function in test_categories:
                print(f"\n📋 TESTING: {category_name.upper().replace('_', ' ')}")
                print("-" * 60)
                
                try:
                    category_result = await test_function(feedback_system)
                    overall_results["test_categories"][category_name] = category_result
                    
                    if category_result.get("success", False):
                        successful_categories += 1
                        print(f"✅ {category_name}: PASSED")
                    else:
                        print(f"❌ {category_name}: FAILED - {category_result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    logger.error(f"Test category {category_name} failed: {e}")
                    overall_results["test_categories"][category_name] = {
                        "success": False,
                        "error": str(e),
                        "category": category_name
                    }
                    print(f"❌ {category_name}: EXCEPTION - {str(e)}")
        
        # Calculate summary
        overall_results["summary"] = {
            "total_test_categories": total_categories,
            "successful_categories": successful_categories,
            "success_rate": successful_categories / total_categories,
            "total_test_time": time.time() - self.start_time,
            "overall_status": "PASSED" if successful_categories == total_categories else "PARTIAL" if successful_categories > 0 else "FAILED"
        }
        
        # Print final summary
        self._print_final_summary(overall_results)
        
        return overall_results
    
    async def test_system_connectivity(self, feedback_system: TetrixFeedbackLoopSystem) -> Dict[str, Any]:
        """Test system connectivity and component health"""
        
        print("Testing system connectivity...")
        
        try:
            connectivity_result = await feedback_system.test_system_connectivity()
            
            analytics_connected = connectivity_result["analytics_service"].get("connected", False)
            llm_available = connectivity_result["llm_engine"].get("available", False)
            overall_status = connectivity_result["overall_status"]
            
            print(f"   Analytics Service: {'✅ Connected' if analytics_connected else '❌ Disconnected'}")
            print(f"   LLM Engine: {'✅ Available' if llm_available else '❌ Unavailable'}")
            print(f"   Overall Status: {overall_status.upper()}")
            
            return {
                "success": overall_status in ["ready", "partial"],
                "connectivity_result": connectivity_result,
                "analytics_connected": analytics_connected,
                "llm_available": llm_available
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def test_abry_partners_documents(self, feedback_system: TetrixFeedbackLoopSystem) -> Dict[str, Any]:
        """Test with Abry Partners documents mentioned in the meeting"""
        
        print("Testing Abry Partners documents from meeting examples...")
        
        try:
            test_documents = self.test_data_generator.get_abry_partners_test_documents()
            results = []
            
            for i, test_doc in enumerate(test_documents):
                print(f"   Processing {test_doc['name']}...")
                
                start_time = time.time()
                result = await feedback_system.run_feedback_loop(
                    extracted_document=test_doc["extracted_document"],
                    document_path=test_doc["document_path"],
                    client_entity_or_org=test_doc["client_entity"],
                    ce_or_org_id=test_doc["ce_or_org_id"]
                )
                processing_time = time.time() - start_time
                
                # Evaluate against ground truth
                accuracy = self._calculate_accuracy(result.improved_document, test_doc["ground_truth"])
                
                test_result = {
                    "document_name": test_doc["name"],
                    "processing_successful": result.feedback_loop_successful,
                    "accuracy": accuracy,
                    "processing_time": processing_time,
                    "corrections_applied": result.processing_results.get("summary", {}).get("corrections_applied", 0),
                    "issues_found": len(result.analytics_before.discrepancies) + len(result.analytics_before.focus_points),
                    "improvement_score": result.improvement_metrics.get("improvement_score", 0)
                }
                
                results.append(test_result)
                
                print(f"      ✅ Accuracy: {accuracy:.1%}, Corrections: {test_result['corrections_applied']}, Time: {processing_time:.2f}s")
            
            # Calculate overall performance
            avg_accuracy = sum(r["accuracy"] for r in results) / len(results)
            total_corrections = sum(r["corrections_applied"] for r in results)
            
            return {
                "success": avg_accuracy >= 0.75,  # 75% accuracy threshold
                "test_results": results,
                "summary": {
                    "documents_tested": len(results),
                    "average_accuracy": avg_accuracy,
                    "total_corrections_applied": total_corrections,
                    "all_processed_successfully": all(r["processing_successful"] for r in results)
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def test_complex_error_handling(self, feedback_system: TetrixFeedbackLoopSystem) -> Dict[str, Any]:
        """Test handling of complex, multi-error documents"""
        
        print("Testing complex error handling...")
        
        try:
            complex_documents = self.test_data_generator.get_complex_error_documents()
            results = []
            
            for test_doc in complex_documents:
                print(f"   Processing {test_doc['name']}...")
                
                result = await feedback_system.run_feedback_loop(
                    extracted_document=test_doc["extracted_document"],
                    document_path=test_doc["document_path"],
                    client_entity_or_org=test_doc["client_entity"],
                    ce_or_org_id=test_doc["ce_or_org_id"]
                )
                
                # Evaluate error handling capability
                initial_errors = len(test_doc["expected_discrepancies"]) + len(test_doc["expected_focus_points"])
                corrections_applied = result.processing_results.get("summary", {}).get("corrections_applied", 0)
                
                error_resolution_rate = corrections_applied / max(initial_errors, 1)
                
                test_result = {
                    "document_name": test_doc["name"],
                    "initial_errors": initial_errors,
                    "corrections_applied": corrections_applied,
                    "error_resolution_rate": error_resolution_rate,
                    "processing_successful": result.feedback_loop_successful
                }
                
                results.append(test_result)
                
                print(f"      ✅ Error Resolution: {error_resolution_rate:.1%}, Corrections: {corrections_applied}")
            
            avg_resolution_rate = sum(r["error_resolution_rate"] for r in results) / len(results)
            
            return {
                "success": avg_resolution_rate >= 0.60,  # 60% error resolution threshold
                "test_results": results,
                "avg_error_resolution_rate": avg_resolution_rate
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def test_batch_processing(self, feedback_system: TetrixFeedbackLoopSystem) -> Dict[str, Any]:
        """Test batch processing capabilities"""
        
        print("Testing batch processing...")
        
        try:
            # Get a mix of documents for batch testing
            abry_docs = self.test_data_generator.get_abry_partners_test_documents()[:2]
            complex_docs = self.test_data_generator.get_complex_error_documents()[:1]
            
            all_docs = abry_docs + complex_docs
            
            documents = [doc["extracted_document"] for doc in all_docs]
            document_paths = [doc["document_path"] for doc in all_docs]
            
            print(f"   Processing {len(documents)} documents in batch...")
            
            start_time = time.time()
            batch_results = await feedback_system.batch_process_documents(
                documents, document_paths
            )
            batch_time = time.time() - start_time
            
            successful_processing = sum(1 for r in batch_results if r.feedback_loop_successful)
            total_corrections = sum(r.processing_results.get("summary", {}).get("corrections_applied", 0) for r in batch_results)
            
            print(f"      ✅ Batch Success: {successful_processing}/{len(documents)}, Total Corrections: {total_corrections}, Time: {batch_time:.2f}s")
            
            return {
                "success": successful_processing >= len(documents) * 0.8,  # 80% success rate
                "documents_processed": len(documents),
                "successful_processing": successful_processing,
                "total_corrections": total_corrections,
                "batch_processing_time": batch_time,
                "avg_processing_time_per_doc": batch_time / len(documents)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def test_performance_metrics(self, feedback_system: TetrixFeedbackLoopSystem) -> Dict[str, Any]:
        """Test performance metrics and system monitoring"""
        
        print("Testing performance metrics...")
        
        try:
            # Process a test document to generate metrics
            test_doc = self.test_data_generator.get_abry_partners_test_documents()[0]
            
            await feedback_system.run_feedback_loop(
                extracted_document=test_doc["extracted_document"],
                document_path=test_doc["document_path"]
            )
            
            # Get system status
            system_status = feedback_system.get_system_status()
            
            print(f"   ✅ System Metrics Retrieved")
            print(f"      Documents Processed: {system_status['system_metrics']['documents_processed']}")
            print(f"      Total Issues Found: {system_status['system_metrics']['total_issues_found']}")
            print(f"      Total Corrections: {system_status['system_metrics']['total_corrections_applied']}")
            print(f"      Avg Improvement: {system_status['system_metrics']['avg_improvement_score']:.1%}")
            
            return {
                "success": True,
                "system_status": system_status,
                "metrics_available": all([
                    "system_metrics" in system_status,
                    "analytics_client_metrics" in system_status,
                    "issue_processor_stats" in system_status
                ])
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def test_end_to_end_workflow(self, feedback_system: TetrixFeedbackLoopSystem) -> Dict[str, Any]:
        """Test the complete end-to-end workflow as described in the meeting"""
        
        print("Testing end-to-end workflow simulation...")
        
        try:
            # 1. Document extraction → 2. Analytics API → 3. AI processing → 4. Improvement
            
            test_doc = self.test_data_generator.get_abry_partners_test_documents()[0]
            
            print("   Step 1: Simulating document extraction...")
            extracted_document = test_doc["extracted_document"]
            
            print("   Step 2: Calling Grant's analytics API...")
            analytics_response = await feedback_system.analytics_client.get_discrepancies_for_document(
                doc_path=test_doc["document_path"]
            )
            
            print(f"      Found {len(analytics_response.discrepancies)} discrepancies, {len(analytics_response.focus_points)} focus points")
            
            print("   Step 3: Processing with AI feedback loop...")
            feedback_result = await feedback_system.run_feedback_loop(
                extracted_document=extracted_document,
                document_path=test_doc["document_path"]
            )
            
            print("   Step 4: Measuring improvement...")
            accuracy = self._calculate_accuracy(feedback_result.improved_document, test_doc["ground_truth"])
            
            workflow_success = (
                feedback_result.feedback_loop_successful and
                accuracy >= 0.80 and
                feedback_result.processing_results.get("summary", {}).get("corrections_applied", 0) > 0
            )
            
            print(f"   ✅ End-to-End Workflow: {'SUCCESS' if workflow_success else 'NEEDS IMPROVEMENT'}")
            print(f"      Final Accuracy: {accuracy:.1%}")
            print(f"      Corrections Applied: {feedback_result.processing_results.get('summary', {}).get('corrections_applied', 0)}")
            
            return {
                "success": workflow_success,
                "accuracy": accuracy,
                "corrections_applied": feedback_result.processing_results.get("summary", {}).get("corrections_applied", 0),
                "workflow_stages_completed": 4,
                "processing_time": feedback_result.processing_time
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _calculate_accuracy(self, improved_document: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        """Calculate accuracy against ground truth"""
        if not ground_truth:
            return 1.0
        
        correct_fields = 0
        total_fields = len(ground_truth)
        
        for field, expected_value in ground_truth.items():
            actual_value = improved_document.get(field)
            
            if isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
                tolerance = max(abs(expected_value) * 0.01, 1.0)
                if abs(actual_value - expected_value) <= tolerance:
                    correct_fields += 1
            elif str(actual_value) == str(expected_value):
                correct_fields += 1
        
        return correct_fields / total_fields if total_fields > 0 else 1.0
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """Get environment information for debugging"""
        return {
            "python_version": sys.version,
            "openai_api_key_set": bool(os.getenv("OPENAI_API_KEY")),
            "anthropic_api_key_set": bool(os.getenv("ANTHROPIC_API_KEY")),
            "tetrix_analytics_url": os.getenv("TETRIX_ANALYTICS_URL", "default"),
            "use_mock_analytics": os.getenv("USE_MOCK_ANALYTICS", "auto"),
            "timestamp": datetime.now().isoformat()
        }
    
    def _print_final_summary(self, results: Dict[str, Any]):
        """Print comprehensive test summary"""
        print(f"\n{'=' * 80}")
        print("🏁 TETRIX AI FEEDBACK LOOP SYSTEM - TEST RESULTS SUMMARY")
        print(f"{'=' * 80}")
        
        summary = results["summary"]
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   Status: {summary['overall_status']}")
        print(f"   Success Rate: {summary['success_rate']:.1%}")
        print(f"   Test Categories: {summary['successful_categories']}/{summary['total_test_categories']}")
        print(f"   Total Test Time: {summary['total_test_time']:.2f}s")
        
        print(f"\n🔍 DETAILED RESULTS:")
        for category_name, category_result in results["test_categories"].items():
            status = "✅ PASSED" if category_result.get("success", False) else "❌ FAILED"
            print(f"   {category_name.replace('_', ' ').title()}: {status}")
            
            if not category_result.get("success", False) and "error" in category_result:
                print(f"      Error: {category_result['error']}")
        
        print(f"\n🎯 SYSTEM ASSESSMENT:")
        if summary["success_rate"] >= 1.0:
            print("   🌟 EXCELLENT: All tests passed! System ready for production deployment.")
            print("   The AI feedback loop demonstrates strong real-world performance.")
        elif summary["success_rate"] >= 0.8:
            print("   ✅ GOOD: Most tests passed. System shows strong potential.")
            print("   Minor issues should be addressed before full deployment.")
        elif summary["success_rate"] >= 0.6:
            print("   ⚠️  PARTIAL: Some tests passed. System needs refinement.")
            print("   Additional development and testing recommended.")
        else:
            print("   ❌ NEEDS WORK: Major issues detected. System requires significant development.")
            print("   Address fundamental issues before proceeding.")
        
        print(f"\n📈 NEXT STEPS:")
        print("   1. Review detailed test results for specific issues")
        print("   2. Address any failed test categories")
        print("   3. Validate with additional real-world documents")
        print("   4. Monitor performance in production environment")
        print("   5. Set up continuous evaluation with Grant's analytics system")

async def main():
    """Main test runner"""
    
    # Check command line arguments
    use_mock = "--mock" in sys.argv
    verbose = "--verbose" in sys.argv
    
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("Starting Tetrix AI Feedback Loop System Test Suite...")
    print(f"Mock Analytics Mode: {use_mock}")
    
    # Run the comprehensive test suite
    test_suite = TetrixTestSuite()
    results = await test_suite.run_all_tests(use_mock_analytics=use_mock)
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"tetrix_test_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Test results saved to: {results_file}")
    
    # Exit with appropriate code
    success_rate = results["summary"]["success_rate"]
    exit_code = 0 if success_rate >= 0.8 else 1
    
    print(f"\nExiting with code {exit_code}")
    return exit_code

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)