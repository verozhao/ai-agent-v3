"""
Accuracy Measurement and Evaluation System for Tetrix AI Feedback Loop
Advanced evaluation metrics, JSON diffing, and performance analysis
Based on Stefan's evaluation framework mentioned in the meeting
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio
import statistics

from feedback_loop_system import TetrixFeedbackLoopSystem, FeedbackLoopResult
from analytics_client import AnalyticsResponse

logger = logging.getLogger(__name__)

@dataclass
class FieldComparison:
    """Detailed comparison of a single field"""
    field_name: str
    expected_value: Any
    actual_value: Any
    is_correct: bool
    error_type: str  # 'missing', 'incorrect', 'type_mismatch', 'precision_error'
    precision_used: Optional[float] = None
    confidence: float = 1.0

@dataclass
class DocumentComparison:
    """Comprehensive comparison between two documents"""
    document_id: str
    total_fields: int
    correct_fields: int
    accuracy: float
    field_comparisons: List[FieldComparison]
    missing_fields: List[str]
    extra_fields: List[str]
    error_summary: Dict[str, int]
    comparison_timestamp: datetime

@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for the feedback loop system"""
    total_documents: int
    successful_processing: int
    avg_accuracy: float
    avg_improvement_score: float
    avg_processing_time: float
    total_corrections_applied: int
    total_issues_resolved: int
    error_distribution: Dict[str, int]
    performance_by_document_type: Dict[str, Dict[str, float]]
    temporal_performance: List[Dict[str, Any]]
    confidence_intervals: Dict[str, Tuple[float, float]]

class AdvancedJsonDiffer:
    """
    Advanced JSON diffing capabilities similar to Stefan's framework
    Handles complex financial document structures with precision requirements
    """
    
    def __init__(self, numeric_tolerance: float = 0.01, percentage_tolerance: float = 0.001):
        self.numeric_tolerance = numeric_tolerance  # 1% default tolerance
        self.percentage_tolerance = percentage_tolerance  # 0.1% for percentages
        self.field_type_mappings = self._initialize_field_mappings()
    
    def _initialize_field_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Initialize field-specific comparison rules"""
        return {
            "currency_fields": {
                "patterns": ["amount", "value", "size", "nav", "proceeds", "commitments"],
                "tolerance_type": "absolute",
                "tolerance": 0.01  # 1% tolerance for currency
            },
            "percentage_fields": {
                "patterns": ["irr", "rate", "percentage", "%"],
                "tolerance_type": "absolute", 
                "tolerance": 0.001  # 0.1% tolerance for percentages
            },
            "date_fields": {
                "patterns": ["date", "timestamp"],
                "tolerance_type": "exact",
                "tolerance": 0
            },
            "count_fields": {
                "patterns": ["count", "number", "quantity"],
                "tolerance_type": "exact",
                "tolerance": 0
            },
            "identifier_fields": {
                "patterns": ["id", "name", "fund_name"],
                "tolerance_type": "exact",
                "tolerance": 0
            }
        }
    
    def compare_documents(self, expected: Dict[str, Any], actual: Dict[str, Any], 
                         document_id: str = "unknown") -> DocumentComparison:
        """
        Perform comprehensive document comparison with financial domain intelligence
        """
        
        all_fields = set(expected.keys()) | set(actual.keys())
        field_comparisons = []
        correct_fields = 0
        
        missing_fields = [f for f in expected.keys() if f not in actual]
        extra_fields = [f for f in actual.keys() if f not in expected]
        
        error_summary = {
            "missing": 0,
            "incorrect": 0,
            "type_mismatch": 0,
            "precision_error": 0,
            "exact_match": 0
        }
        
        for field in expected.keys():
            if field not in actual:
                # Missing field
                comparison = FieldComparison(
                    field_name=field,
                    expected_value=expected[field],
                    actual_value=None,
                    is_correct=False,
                    error_type="missing",
                    confidence=1.0
                )
                error_summary["missing"] += 1
            else:
                # Compare existing field
                comparison = self._compare_field(field, expected[field], actual[field])
                if comparison.is_correct:
                    correct_fields += 1
                    error_summary["exact_match"] += 1
                else:
                    error_summary[comparison.error_type] += 1
            
            field_comparisons.append(comparison)
        
        # Handle extra fields in actual document
        for field in extra_fields:
            comparison = FieldComparison(
                field_name=field,
                expected_value=None,
                actual_value=actual[field],
                is_correct=False,
                error_type="extra_field",
                confidence=0.5  # Lower confidence for extra fields
            )
            field_comparisons.append(comparison)
        
        accuracy = correct_fields / len(expected) if expected else 1.0
        
        return DocumentComparison(
            document_id=document_id,
            total_fields=len(expected),
            correct_fields=correct_fields,
            accuracy=accuracy,
            field_comparisons=field_comparisons,
            missing_fields=missing_fields,
            extra_fields=extra_fields,
            error_summary=error_summary,
            comparison_timestamp=datetime.now()
        )
    
    def _compare_field(self, field_name: str, expected: Any, actual: Any) -> FieldComparison:
        """Compare individual field with domain-specific logic"""
        
        # Determine field type and comparison rules
        field_config = self._get_field_config(field_name)
        tolerance_type = field_config["tolerance_type"]
        tolerance = field_config["tolerance"]
        
        # Handle None values
        if expected is None and actual is None:
            return FieldComparison(
                field_name=field_name,
                expected_value=expected,
                actual_value=actual,
                is_correct=True,
                error_type="exact_match"
            )
        
        if expected is None or actual is None:
            return FieldComparison(
                field_name=field_name,
                expected_value=expected,
                actual_value=actual,
                is_correct=False,
                error_type="missing"
            )
        
        # Type checking
        if type(expected) != type(actual):
            # Try to convert if possible
            try:
                if isinstance(expected, (int, float)) and isinstance(actual, str):
                    actual = float(actual.replace('%', '').replace(',', ''))
                elif isinstance(actual, (int, float)) and isinstance(expected, str):
                    expected = float(expected.replace('%', '').replace(',', ''))
                else:
                    return FieldComparison(
                        field_name=field_name,
                        expected_value=expected,
                        actual_value=actual,
                        is_correct=False,
                        error_type="type_mismatch"
                    )
            except:
                return FieldComparison(
                    field_name=field_name,
                    expected_value=expected,
                    actual_value=actual,
                    is_correct=False,
                    error_type="type_mismatch"
                )
        
        # Numeric comparison with tolerance
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            if tolerance_type == "exact":
                is_correct = expected == actual
                error_type = "exact_match" if is_correct else "incorrect"
            else:
                # Apply tolerance
                if tolerance_type == "percentage":
                    tolerance_value = abs(expected) * tolerance
                else:  # absolute
                    tolerance_value = tolerance
                
                is_correct = abs(expected - actual) <= tolerance_value
                error_type = "exact_match" if is_correct else "precision_error"
                
            return FieldComparison(
                field_name=field_name,
                expected_value=expected,
                actual_value=actual,
                is_correct=is_correct,
                error_type=error_type,
                precision_used=tolerance if not is_correct else None
            )
        
        # String comparison
        elif isinstance(expected, str) and isinstance(actual, str):
            is_correct = expected.strip().lower() == actual.strip().lower()
            return FieldComparison(
                field_name=field_name,
                expected_value=expected,
                actual_value=actual,
                is_correct=is_correct,
                error_type="exact_match" if is_correct else "incorrect"
            )
        
        # Default exact comparison
        else:
            is_correct = expected == actual
            return FieldComparison(
                field_name=field_name,
                expected_value=expected,
                actual_value=actual,
                is_correct=is_correct,
                error_type="exact_match" if is_correct else "incorrect"
            )
    
    def _get_field_config(self, field_name: str) -> Dict[str, Any]:
        """Get field-specific configuration based on name patterns"""
        
        field_lower = field_name.lower()
        
        for field_type, config in self.field_type_mappings.items():
            for pattern in config["patterns"]:
                if pattern in field_lower:
                    return config
        
        # Default configuration
        return {
            "tolerance_type": "exact",
            "tolerance": 0
        }

class EvaluationSystem:
    """
    Comprehensive evaluation system for the Tetrix AI feedback loop
    Provides detailed performance analysis and improvement tracking
    """
    
    def __init__(self):
        self.json_differ = AdvancedJsonDiffer()
        self.evaluation_history = []
        self.performance_trends = {}
        
    async def evaluate_feedback_loop_performance(self, 
                                               feedback_system: TetrixFeedbackLoopSystem,
                                               test_documents: List[Dict[str, Any]],
                                               ground_truths: List[Dict[str, Any]],
                                               document_paths: List[str]) -> EvaluationMetrics:
        """
        Comprehensive evaluation of feedback loop performance
        Similar to the evaluation framework Stefan built
        """
        
        if len(test_documents) != len(ground_truths) or len(test_documents) != len(document_paths):
            raise ValueError("All input lists must have the same length")
        
        logger.info(f"Starting comprehensive evaluation of {len(test_documents)} documents")
        
        evaluation_start = time.time()
        document_comparisons = []
        processing_times = []
        improvement_scores = []
        corrections_applied = []
        issues_resolved = []
        
        performance_by_type = {}
        temporal_performance = []
        
        for i, (test_doc, ground_truth, doc_path) in enumerate(zip(test_documents, ground_truths, document_paths)):
            logger.info(f"Evaluating document {i+1}/{len(test_documents)}: {doc_path}")
            
            doc_start_time = time.time()
            
            try:
                # Run feedback loop
                feedback_result = await feedback_system.run_feedback_loop(
                    extracted_document=test_doc,
                    document_path=doc_path
                )
                
                processing_time = time.time() - doc_start_time
                processing_times.append(processing_time)
                
                if feedback_result.feedback_loop_successful:
                    # Compare improved document with ground truth
                    comparison = self.json_differ.compare_documents(
                        expected=ground_truth,
                        actual=feedback_result.improved_document,
                        document_id=feedback_result.document_id
                    )
                    
                    document_comparisons.append(comparison)
                    
                    # Extract metrics
                    improvement_score = feedback_result.improvement_metrics.get("improvement_score", 0)
                    improvement_scores.append(improvement_score)
                    
                    corrections = feedback_result.processing_results.get("summary", {}).get("corrections_applied", 0)
                    corrections_applied.append(corrections)
                    
                    issues_before = len(feedback_result.analytics_before.discrepancies) + len(feedback_result.analytics_before.focus_points)
                    issues_after = 0
                    if feedback_result.analytics_after:
                        issues_after = len(feedback_result.analytics_after.discrepancies) + len(feedback_result.analytics_after.focus_points)
                    
                    resolved = max(0, issues_before - issues_after)
                    issues_resolved.append(resolved)
                    
                    # Track performance by document type
                    doc_type = feedback_result.analytics_before.document_type
                    if doc_type not in performance_by_type:
                        performance_by_type[doc_type] = []
                    performance_by_type[doc_type].append({
                        "accuracy": comparison.accuracy,
                        "improvement_score": improvement_score,
                        "processing_time": processing_time
                    })
                    
                    # Temporal performance tracking
                    temporal_performance.append({
                        "timestamp": datetime.now().isoformat(),
                        "document_index": i,
                        "accuracy": comparison.accuracy,
                        "processing_time": processing_time,
                        "corrections_applied": corrections
                    })
                
            except Exception as e:
                logger.error(f"Evaluation failed for document {i}: {e}")
                continue
        
        # Calculate comprehensive metrics
        total_evaluation_time = time.time() - evaluation_start
        
        # Basic statistics
        successful_processing = len(document_comparisons)
        avg_accuracy = statistics.mean([comp.accuracy for comp in document_comparisons]) if document_comparisons else 0.0
        avg_improvement = statistics.mean(improvement_scores) if improvement_scores else 0.0
        avg_processing_time = statistics.mean(processing_times) if processing_times else 0.0
        
        # Error distribution analysis
        error_distribution = {}
        for comparison in document_comparisons:
            for error_type, count in comparison.error_summary.items():
                if error_type not in error_distribution:
                    error_distribution[error_type] = 0
                error_distribution[error_type] += count
        
        # Performance by document type
        performance_by_type_summary = {}
        for doc_type, performances in performance_by_type.items():
            performance_by_type_summary[doc_type] = {
                "avg_accuracy": statistics.mean([p["accuracy"] for p in performances]),
                "avg_improvement": statistics.mean([p["improvement_score"] for p in performances]),
                "avg_processing_time": statistics.mean([p["processing_time"] for p in performances]),
                "document_count": len(performances)
            }
        
        # Confidence intervals (95%)
        confidence_intervals = {}
        if len(document_comparisons) > 1:
            accuracies = [comp.accuracy for comp in document_comparisons]
            processing_times_list = processing_times
            
            confidence_intervals = {
                "accuracy": self._calculate_confidence_interval(accuracies),
                "processing_time": self._calculate_confidence_interval(processing_times_list),
                "improvement_score": self._calculate_confidence_interval(improvement_scores) if improvement_scores else (0.0, 0.0)
            }
        
        evaluation_metrics = EvaluationMetrics(
            total_documents=len(test_documents),
            successful_processing=successful_processing,
            avg_accuracy=avg_accuracy,
            avg_improvement_score=avg_improvement,
            avg_processing_time=avg_processing_time,
            total_corrections_applied=sum(corrections_applied),
            total_issues_resolved=sum(issues_resolved),
            error_distribution=error_distribution,
            performance_by_document_type=performance_by_type_summary,
            temporal_performance=temporal_performance,
            confidence_intervals=confidence_intervals
        )
        
        # Store evaluation results
        evaluation_record = {
            "timestamp": datetime.now().isoformat(),
            "metrics": asdict(evaluation_metrics),
            "document_comparisons": [asdict(comp) for comp in document_comparisons],
            "total_evaluation_time": total_evaluation_time
        }
        
        self.evaluation_history.append(evaluation_record)
        
        logger.info(f"Evaluation completed: {avg_accuracy:.1%} accuracy, {avg_improvement:.1%} improvement, {avg_processing_time:.2f}s avg processing")
        
        return evaluation_metrics
    
    def _calculate_confidence_interval(self, values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for a list of values"""
        if len(values) < 2:
            return (0.0, 0.0)
        
        mean = statistics.mean(values)
        stdev = statistics.stdev(values)
        n = len(values)
        
        # Using t-distribution approximation
        t_score = 1.96  # 95% confidence for large n
        margin = t_score * (stdev / (n ** 0.5))
        
        return (mean - margin, mean + margin)
    
    def analyze_improvement_over_time(self) -> Dict[str, Any]:
        """Analyze how the system performance improves over time"""
        
        if len(self.evaluation_history) < 2:
            return {"message": "Insufficient data for trend analysis"}
        
        accuracy_trend = []
        improvement_trend = []
        processing_time_trend = []
        
        for record in self.evaluation_history:
            metrics = record["metrics"]
            accuracy_trend.append(metrics["avg_accuracy"])
            improvement_trend.append(metrics["avg_improvement_score"])
            processing_time_trend.append(metrics["avg_processing_time"])
        
        # Calculate trends
        accuracy_improvement = accuracy_trend[-1] - accuracy_trend[0] if len(accuracy_trend) > 1 else 0
        processing_efficiency_gain = processing_time_trend[0] - processing_time_trend[-1] if len(processing_time_trend) > 1 else 0
        
        return {
            "evaluation_runs": len(self.evaluation_history),
            "accuracy_trend": accuracy_trend,
            "improvement_trend": improvement_trend,
            "processing_time_trend": processing_time_trend,
            "accuracy_improvement": accuracy_improvement,
            "processing_efficiency_gain": processing_efficiency_gain,
            "is_improving": accuracy_improvement > 0,
            "trend_analysis_timestamp": datetime.now().isoformat()
        }
    
    def generate_detailed_report(self, evaluation_metrics: EvaluationMetrics, 
                               include_recommendations: bool = True) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        
        report = {
            "executive_summary": {
                "overall_performance": self._assess_overall_performance(evaluation_metrics),
                "key_metrics": {
                    "accuracy": f"{evaluation_metrics.avg_accuracy:.1%}",
                    "improvement_score": f"{evaluation_metrics.avg_improvement_score:.1%}",
                    "processing_time": f"{evaluation_metrics.avg_processing_time:.2f}s",
                    "success_rate": f"{evaluation_metrics.successful_processing / evaluation_metrics.total_documents:.1%}"
                },
                "total_impact": {
                    "documents_processed": evaluation_metrics.total_documents,
                    "corrections_applied": evaluation_metrics.total_corrections_applied,
                    "issues_resolved": evaluation_metrics.total_issues_resolved
                }
            },
            "detailed_analysis": {
                "accuracy_distribution": self._analyze_accuracy_distribution(evaluation_metrics),
                "error_analysis": self._analyze_error_patterns(evaluation_metrics),
                "performance_by_document_type": evaluation_metrics.performance_by_document_type,
                "confidence_intervals": evaluation_metrics.confidence_intervals
            },
            "temporal_analysis": {
                "performance_over_time": evaluation_metrics.temporal_performance,
                "trend_analysis": self.analyze_improvement_over_time()
            },
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "evaluation_framework_version": "1.0",
                "total_evaluations_conducted": len(self.evaluation_history)
            }
        }
        
        if include_recommendations:
            report["recommendations"] = self._generate_recommendations(evaluation_metrics)
        
        return report
    
    def _assess_overall_performance(self, metrics: EvaluationMetrics) -> str:
        """Assess overall system performance"""
        
        accuracy = metrics.avg_accuracy
        improvement = metrics.avg_improvement_score
        success_rate = metrics.successful_processing / metrics.total_documents
        
        if accuracy >= 0.95 and improvement >= 0.80 and success_rate >= 0.95:
            return "EXCELLENT"
        elif accuracy >= 0.85 and improvement >= 0.60 and success_rate >= 0.80:
            return "GOOD"
        elif accuracy >= 0.70 and improvement >= 0.40 and success_rate >= 0.70:
            return "SATISFACTORY"
        else:
            return "NEEDS_IMPROVEMENT"
    
    def _analyze_accuracy_distribution(self, metrics: EvaluationMetrics) -> Dict[str, Any]:
        """Analyze distribution of accuracy scores"""
        # This would be implemented with access to individual document accuracies
        return {
            "average_accuracy": metrics.avg_accuracy,
            "accuracy_range": "Would need individual scores for detailed analysis",
            "high_accuracy_documents": ">=90% accuracy",
            "low_accuracy_documents": "<70% accuracy"
        }
    
    def _analyze_error_patterns(self, metrics: EvaluationMetrics) -> Dict[str, Any]:
        """Analyze common error patterns"""
        
        total_errors = sum(metrics.error_distribution.values())
        error_percentages = {}
        
        for error_type, count in metrics.error_distribution.items():
            error_percentages[error_type] = (count / total_errors) * 100 if total_errors > 0 else 0
        
        most_common_error = max(metrics.error_distribution.items(), key=lambda x: x[1])[0] if metrics.error_distribution else "none"
        
        return {
            "total_errors": total_errors,
            "error_distribution": error_percentages,
            "most_common_error_type": most_common_error,
            "error_resolution_rate": (metrics.total_issues_resolved / max(total_errors, 1)) * 100
        }
    
    def _generate_recommendations(self, metrics: EvaluationMetrics) -> List[str]:
        """Generate actionable recommendations based on evaluation results"""
        
        recommendations = []
        
        if metrics.avg_accuracy < 0.85:
            recommendations.append("Consider improving field detection accuracy - accuracy below 85%")
        
        if metrics.avg_processing_time > 5.0:
            recommendations.append("Optimize processing time - currently averaging over 5 seconds per document")
        
        if metrics.total_corrections_applied / metrics.total_documents < 0.5:
            recommendations.append("Review correction application logic - low correction rate detected")
        
        if "missing" in metrics.error_distribution and metrics.error_distribution["missing"] > 0:
            recommendations.append("Address missing field detection - implement fallback strategies")
        
        if metrics.avg_improvement_score < 0.60:
            recommendations.append("Enhance improvement algorithms - current improvement score below target")
        
        success_rate = metrics.successful_processing / metrics.total_documents
        if success_rate < 0.90:
            recommendations.append("Improve system reliability - success rate below 90%")
        
        if not recommendations:
            recommendations.append("System performing well - continue monitoring and minor optimizations")
        
        return recommendations
    
    def export_evaluation_results(self, filepath: str, include_detailed_comparisons: bool = False):
        """Export evaluation results to file"""
        
        export_data = {
            "evaluation_history": self.evaluation_history,
            "performance_trends": self.performance_trends,
            "export_timestamp": datetime.now().isoformat(),
            "export_version": "1.0"
        }
        
        if include_detailed_comparisons:
            # Include detailed document comparisons
            export_data["detailed_comparisons"] = "Would include full comparison data"
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Evaluation results exported to {filepath}")

# Export
__all__ = ["EvaluationSystem", "EvaluationMetrics", "AdvancedJsonDiffer", "DocumentComparison", "FieldComparison"]