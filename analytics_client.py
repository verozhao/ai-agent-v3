"""
Fixed Analytics Microservice Client
Proper integration with Grant's tetrix-analytics-microservice endpoints
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import aiohttp
import os

logger = logging.getLogger(__name__)

@dataclass
class Discrepancy:
    """Represents a mathematical inconsistency in financial data"""
    discrepancy_id: str
    field: str
    issue_type: str
    current_value: Any
    expected_value: Any
    confidence: float
    message: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    financial_rule: str
    evidence: List[str]

@dataclass
class FocusPoint:
    """Represents suspicious data that should be reviewed"""
    focus_point_id: str
    field: str
    issue_type: str
    current_value: Any
    flag_reason: str
    confidence: float
    message: str
    historical_values: List[Any]
    comparison_context: Dict[str, Any]

@dataclass
class AnalyticsResponse:
    """Response from analytics microservice"""
    document_path: str
    document_type: str
    discrepancies: List[Discrepancy]
    focus_points: List[FocusPoint]
    consolidation_metadata: Dict[str, Any]
    historical_data: Dict[str, Any]
    response_timestamp: datetime
    processing_time_ms: float

class TetrixAnalyticsClient:
    """Client for Grant's tetrix-analytics-microservice"""
    
    def __init__(self, base_url: str = None, vpn_required: bool = True):
        # From meeting: http://internal-backen-micro-f3m5zasfnrzz-435617696.us-east-2.elb.amazonaws.com
        self.base_url = base_url or "http://internal-backen-micro-f3m5zasfnrzz-435617696.us-east-2.elb.amazonaws.com"
        self.vpn_required = vpn_required
        self.session = None
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0,
            "last_connection_test": None
        }
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "tetrix-ai-feedback-loop/1.0"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to analytics microservice"""
        try:
            start_time = time.time()
            
            # Test heartbeat endpoint (no microservice prefix)
            async with self.session.get(f"{self.base_url}/heartbeat") as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    self.metrics["last_connection_test"] = datetime.now().isoformat()
                    return {
                        "connected": True,
                        "status_code": response.status,
                        "response_time": response_time,
                        "vpn_status": "connected" if self.vpn_required else "not_required",
                        "message": "Successfully connected to tetrix-analytics-microservice"
                    }
                else:
                    return {
                        "connected": False,
                        "status_code": response.status,
                        "response_time": response_time,
                        "error": f"Unexpected status code: {response.status}"
                    }
                    
        except aiohttp.ClientConnectorError as e:
            return {
                "connected": False,
                "error": f"Connection failed: {str(e)}",
                "vpn_check": "Ensure VPN is connected" if self.vpn_required else "Check network connectivity"
            }
        except Exception as e:
            return {
                "connected": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    async def get_discrepancies_for_document(self, doc_path: str, client_entity_or_org: str = "client_entity", 
                                           ce_or_org_id: str = "default") -> AnalyticsResponse:
        """
        Get discrepancies and focus points for a specific document
        Try multiple endpoint approaches based on Grant's API
        """
        start_time = time.time()
        
        # Different endpoint strategies to try
        strategies = [
            {
                "name": "tasks_todo_get",
                "method": "GET",
                "endpoint": f"/tetrix-analytics-microservice/discrepancies/tasks_todo/{client_entity_or_org}/{ce_or_org_id}",
                "params": {"doc_path": doc_path},
                "body": None
            },
            {
                "name": "tasks_todo_post",
                "method": "POST", 
                "endpoint": f"/tetrix-analytics-microservice/discrepancies/tasks_todo/{client_entity_or_org}/{ce_or_org_id}",
                "params": {},
                "body": {"doc_path": doc_path}
            },
            {
                "name": "extraction_flags",
                "method": "GET",
                "endpoint": f"/tetrix-analytics-microservice/discrepancies/extraction_flags/{doc_path}",
                "params": {},
                "body": None
            },
            {
                "name": "flag_discrepancies_post",
                "method": "POST",
                "endpoint": f"/tetrix-analytics-microservice/discrepancies/flag_discrepancies_investor/{client_entity_or_org}",
                "params": {},
                "body": {"doc_path": doc_path}
            }
        ]
        
        for strategy in strategies:
            try:
                logger.info(f"Trying strategy: {strategy['name']}")
                url = f"{self.base_url}{strategy['endpoint']}"
                
                self.metrics["total_requests"] += 1
                
                if strategy["method"] == "GET":
                    async with self.session.get(url, params=strategy["params"]) as response:
                        result = await self._handle_response(response, doc_path, start_time, strategy["name"])
                        if result:
                            return result
                
                elif strategy["method"] == "POST":
                    async with self.session.post(url, json=strategy["body"], params=strategy["params"]) as response:
                        result = await self._handle_response(response, doc_path, start_time, strategy["name"])
                        if result:
                            return result
                        
            except Exception as e:
                logger.warning(f"Strategy {strategy['name']} failed: {e}")
                continue
        
        # All strategies failed, return empty response
        logger.error(f"All analytics endpoint strategies failed for {doc_path}")
        self.metrics["failed_requests"] += 1
        
        return AnalyticsResponse(
            document_path=doc_path,
            document_type="unknown",
            discrepancies=[],
            focus_points=[],
            consolidation_metadata={"error": "all_endpoints_failed"},
            historical_data={},
            response_timestamp=datetime.now(),
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    async def _handle_response(self, response, doc_path: str, start_time: float, strategy_name: str) -> Optional[AnalyticsResponse]:
        """Handle API response and return AnalyticsResponse if successful"""
        
        response_time = time.time() - start_time
        
        if response.status == 200:
            try:
                data = await response.json()
                self.metrics["successful_requests"] += 1
                
                # Parse response into structured format
                analytics_response = self._parse_analytics_response(
                    data, doc_path, response_time * 1000
                )
                
                logger.info(f"SUCCESS with strategy {strategy_name}!")
                logger.info(f"Retrieved {len(analytics_response.discrepancies)} discrepancies and "
                          f"{len(analytics_response.focus_points)} focus points for {doc_path}")
                
                return analytics_response
                
            except Exception as e:
                logger.error(f"Failed to parse response from {strategy_name}: {e}")
                return None
                
        elif response.status in [404, 405, 422]:
            error_text = await response.text()
            logger.info(f"Strategy {strategy_name} not supported: {response.status} - {error_text}")
            return None
            
        else:
            error_text = await response.text()
            logger.warning(f"Strategy {strategy_name} failed: {response.status} - {error_text}")
            return None
    
    def _parse_analytics_response(self, data: Dict[str, Any], doc_path: str, 
                                processing_time: float) -> AnalyticsResponse:
        """Parse raw API response into structured AnalyticsResponse"""
        
        discrepancies = []
        focus_points = []
        
        # Parse the actual API response format from Grant's service
        # The real API returns data_flags with flag_type indicating discrepancy vs focus_point
        data_flags = data.get("data_flags", [])
        
        for flag in data_flags:
            flag_type = flag.get("flag_type", "")
            
            if flag_type == "discrepancy":
                # Convert to our Discrepancy format
                discrepancy = Discrepancy(
                    discrepancy_id=flag.get("_id", f"disc_{len(discrepancies)}"),
                    field=flag.get("field_name", "unknown"),
                    issue_type=flag.get("discrepancy_type", "mathematical_inconsistency"),
                    current_value=flag.get("field_value"),
                    expected_value=None,  # Not provided in this format
                    confidence=0.95,  # Discrepancies are high confidence
                    message=flag.get("discrepancy_message", "Mathematical inconsistency detected"),
                    severity="critical" if "decreased" in flag.get("discrepancy_message", "") else "high",
                    financial_rule=flag.get("discrepancy_type", "Mathematical consistency"),
                    evidence=[f"Document: {flag.get('document_type', '')}", 
                             f"Field: {flag.get('field_name', '')}"]
                )
                discrepancies.append(discrepancy)
                
            elif flag_type == "focus_point":
                # Convert to our FocusPoint format
                focus_point = FocusPoint(
                    focus_point_id=flag.get("_id", f"fp_{len(focus_points)}"),
                    field=flag.get("field_name", "unknown"),
                    issue_type=flag.get("discrepancy_type", "suspicious_data"),
                    current_value=flag.get("field_value"),
                    flag_reason=flag.get("discrepancy_message", "Unusual data pattern"),
                    confidence=0.75,  # Focus points are medium confidence
                    message=flag.get("discrepancy_message", "Data requires review"),
                    historical_values=[],  # Not provided in this format
                    comparison_context={"fund_name": flag.get("fund_name", ""), 
                                      "reporting_date": flag.get("reporting_date", "")}
                )
                focus_points.append(focus_point)
        
        # Fallback: try old format for backwards compatibility
        if not data_flags:
            # Parse discrepancies (mathematical inconsistencies)
            raw_discrepancies = data.get("discrepancies", [])
            for i, disc in enumerate(raw_discrepancies):
                discrepancy = Discrepancy(
                    discrepancy_id=disc.get("id", f"disc_{i}"),
                    field=disc.get("field", "unknown"),
                    issue_type=disc.get("issue_type", "mathematical_inconsistency"),
                    current_value=disc.get("current_value"),
                    expected_value=disc.get("expected_value"),
                    confidence=disc.get("confidence", 0.95),
                    message=disc.get("message", "Mathematical inconsistency detected"),
                    severity=disc.get("severity", "critical"),
                    financial_rule=disc.get("financial_rule", "Mathematical consistency"),
                    evidence=disc.get("evidence", [])
                )
                discrepancies.append(discrepancy)
            
            # Parse focus points (suspicious data)
            raw_focus_points = data.get("focus_points", [])
            for i, fp in enumerate(raw_focus_points):
                focus_point = FocusPoint(
                    focus_point_id=fp.get("id", f"fp_{i}"),
                    field=fp.get("field", "unknown"),
                    issue_type=fp.get("issue_type", "suspicious_data"),
                    current_value=fp.get("current_value"),
                    flag_reason=fp.get("flag_reason", "Unusual data pattern"),
                    confidence=fp.get("confidence", 0.75),
                    message=fp.get("message", "Data requires review"),
                    historical_values=fp.get("historical_values", []),
                    comparison_context=fp.get("comparison_context", {})
                )
                focus_points.append(focus_point)
        
        return AnalyticsResponse(
            document_path=doc_path,
            document_type=data.get("document_type", "unknown"),
            discrepancies=discrepancies,
            focus_points=focus_points,
            consolidation_metadata=data.get("consolidation_metadata", {}),
            historical_data=data.get("historical_data", {}),
            response_timestamp=datetime.now(),
            processing_time_ms=processing_time
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics"""
        success_rate = 0.0
        if self.metrics["total_requests"] > 0:
            success_rate = self.metrics["successful_requests"] / self.metrics["total_requests"]
        
        return {
            "total_requests": self.metrics["total_requests"],
            "successful_requests": self.metrics["successful_requests"],
            "failed_requests": self.metrics["failed_requests"],
            "success_rate": success_rate,
            "avg_response_time": self.metrics["avg_response_time"],
            "last_connection_test": self.metrics["last_connection_test"]
        }

class MockAnalyticsClient(TetrixAnalyticsClient):
    """Mock client for testing when VPN/analytics service is not available"""
    
    def __init__(self):
        super().__init__(vpn_required=False)
        
    async def test_connection(self) -> Dict[str, Any]:
        """Mock connection test"""
        return {
            "connected": True,
            "status_code": 200,
            "response_time": 0.1,
            "vpn_status": "mock_mode",
            "message": "Mock analytics client - no real connection"
        }
    
    async def get_discrepancies_for_document(self, doc_path: str, client_entity_or_org: str = "client_entity", 
                                           ce_or_org_id: str = "default") -> AnalyticsResponse:
        """Mock discrepancies response for testing"""
        
        # Generate realistic mock discrepancies and focus points
        discrepancies = [
            Discrepancy(
                discrepancy_id="mock_disc_1",
                field="realized_value",
                issue_type="value_decreased",
                current_value=150000,
                expected_value=650000,
                confidence=0.95,
                message="Realized value decreased when it cannot decrease",
                severity="critical",
                financial_rule="Realized value monotonicity",
                evidence=["Previous value: 650000", "Current value: 150000"]
            ),
            Discrepancy(
                discrepancy_id="mock_disc_2",
                field="total_annual_revenue",
                issue_type="calculation_error",
                current_value=195000000,
                expected_value=200000000,
                confidence=0.98,
                message="Total doesn't match sum of quarterly components",
                severity="high",
                financial_rule="Cumulative total consistency",
                evidence=["Q1+Q2+Q3+Q4 = 200000000", "Reported total = 195000000"]
            )
        ]
        
        focus_points = [
            FocusPoint(
                focus_point_id="mock_fp_1",
                field="investment_valuation",
                issue_type="unusual_increase",
                current_value=20000000,
                flag_reason="Value increased by 1000% from previous period",
                confidence=0.75,
                message="Unusually large valuation increase",
                historical_values=[165000, 131000, 165000],
                comparison_context={"previous_value": 165000, "increase_percentage": 1000.0}
            )
        ]
        
        await asyncio.sleep(0.1)  # Simulate API delay
        
        return AnalyticsResponse(
            document_path=doc_path,
            document_type="PE_fund_report",
            discrepancies=discrepancies,
            focus_points=focus_points,
            consolidation_metadata={"mock_mode": True},
            historical_data={"fund_name": "Abry Partners V", "periods": ["2023-12-31", "2023-09-30"]},
            response_timestamp=datetime.now(),
            processing_time_ms=100.0
        )

# Factory function to create appropriate client
def create_analytics_client(use_mock: bool = None) -> TetrixAnalyticsClient:
    """
    Create analytics client - real or mock based on environment
    """
    if use_mock is None:
        # Auto-detect based on environment
        use_mock = not bool(os.getenv("TETRIX_ANALYTICS_URL")) or os.getenv("USE_MOCK_ANALYTICS", "false").lower() == "true"
    
    if use_mock:
        logger.info("Using mock analytics client")
        return MockAnalyticsClient()
    else:
        logger.info("Using real analytics client")
        analytics_url = os.getenv("TETRIX_ANALYTICS_URL", "http://internal-backen-micro-f3m5zasfnrzz-435617696.us-east-2.elb.amazonaws.com")
        return TetrixAnalyticsClient(base_url=analytics_url)

# Export
__all__ = ["TetrixAnalyticsClient", "MockAnalyticsClient", "AnalyticsResponse", "Discrepancy", "FocusPoint", "create_analytics_client"]