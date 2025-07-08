"""
Fixed Analytics Microservice Client
Proper integration with tetrix-analytics-microservice endpoints
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
    """Client for tetrix-analytics-microservice"""
    
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
        
        # Multiple heartbeat endpoints to try
        heartbeat_endpoints = [
            "/heartbeat",
            "/",
            "/health", 
            "/tetrix-analytics-microservice/heartbeat",
            "/tetrix-analytics-microservice/",
            "/tetrix-analytics-microservice/health"
        ]
        
        for endpoint in heartbeat_endpoints:
            try:
                start_time = time.time()
                
                async with self.session.get(f"{self.base_url}{endpoint}") as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        self.metrics["last_connection_test"] = datetime.now().isoformat()
                        return {
                            "connected": True,
                            "status_code": response.status,
                            "response_time": response_time,
                            "vpn_status": "connected" if self.vpn_required else "not_required",
                            "message": f"Successfully connected to tetrix-analytics-microservice via {endpoint}",
                            "endpoint_used": endpoint
                        }
                    
            except aiohttp.ClientConnectorError as e:
                # Network-level failure, don't continue with other endpoints
                return {
                    "connected": False,
                    "error": f"Connection failed: {str(e)}",
                    "vpn_check": "Ensure VPN is connected" if self.vpn_required else "Check network connectivity"
                }
            except Exception as e:
                # Continue trying other endpoints
                continue
        
        # If all heartbeat endpoints fail, test if we can reach the service at all
        # by trying a known working endpoint with a real document path
        try:
            start_time = time.time()
            test_doc_path = "PEFundPortfolioExtraction/67ee89d7ecbb614e1103e533"
            
            async with self.session.get(
                f"{self.base_url}/tetrix-analytics-microservice/discrepancies/extraction_flags/{test_doc_path}"
            ) as response:
                response_time = time.time() - start_time
                
                # 200 means service is working perfectly
                if response.status == 200:
                    self.metrics["last_connection_test"] = datetime.now().isoformat()
                    return {
                        "connected": True,
                        "status_code": response.status,
                        "response_time": response_time,
                        "vpn_status": "connected" if self.vpn_required else "not_required",
                        "message": "Service fully operational via extraction_flags endpoint",
                        "endpoint_used": "extraction_flags",
                        "note": "Heartbeat endpoint not available, but service is working perfectly"
                    }
                elif response.status in [404, 422]:
                    self.metrics["last_connection_test"] = datetime.now().isoformat()
                    return {
                        "connected": True,
                        "status_code": response.status,
                        "response_time": response_time,
                        "vpn_status": "connected" if self.vpn_required else "not_required",
                        "message": "Service reachable but endpoint test returned error",
                        "endpoint_used": "extraction_flags",
                        "note": "Service is accessible but heartbeat endpoint unavailable"
                    }
                
        except aiohttp.ClientConnectorError as e:
            return {
                "connected": False,
                "error": f"Connection failed: {str(e)}",
                "vpn_check": "Ensure VPN is connected" if self.vpn_required else "Check network connectivity"
            }
        except Exception as e:
            pass
        
        # Complete failure
        return {
            "connected": False,
            "error": "All connection attempts failed",
            "endpoints_tested": heartbeat_endpoints,
            "vpn_check": "Ensure VPN is connected" if self.vpn_required else "Check network connectivity"
        }
    
    async def get_discrepancies_for_document(self, doc_path: str, client_entity_or_org: str = "client_entity",
                                           ce_or_org_id: str = "default") -> AnalyticsResponse:
        """
        Get discrepancies and focus points for a specific document
        Try multiple endpoint approaches based on API
        """
        start_time = time.time()
        
        strategies = [
            {
                "name": "extraction_flags",
                "method": "GET",
                "endpoint": f"/tetrix-analytics-microservice/discrepancies/extraction_flags/{doc_path}",
                "params": {},
                "body": None
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
        
        # Parse the actual API response format from service
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
    
    async def revalidate_improved_document(self, original_doc_path: str, improved_document: Dict[str, Any],
                                          client_entity_or_org: str = "client_entity", 
                                          ce_or_org_id: str = "default") -> AnalyticsResponse:
        """
        Re-validate improved document by submitting it for fresh analysis
        This proves whether our corrections actually reduced the number of issues
        
        Note: In a real system, this would submit the improved document to the extraction pipeline
        and get fresh analytics. For now, we simulate this by calling the same endpoint
        which will return the original issues (not ideal but shows the validation concept)
        """
        
        # For the current implementation, we call the same endpoint since we can't 
        # actually submit improved documents to Grant's extraction pipeline yet
        # This is a limitation - ideally we'd have an endpoint to submit improved docs
        
        logger.info(f"Re-validating improved document for {original_doc_path}")
        logger.info("Note: Currently using same endpoint - future enhancement needed to submit improved docs")
        
        # Call the same endpoint (this is the limitation - we need Grant to add an endpoint 
        # that accepts improved documents for re-analysis)
        return await self.get_discrepancies_for_document(
            doc_path=original_doc_path,
            client_entity_or_org=client_entity_or_org,
            ce_or_org_id=ce_or_org_id
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
    
    async def get_raw_document_data(self, document_path: str) -> Dict[str, Any]:
        """Get the raw document data for calculations and analysis using real Tetrix API endpoints"""
        
        logger.info(f"Fetching raw document data for {document_path} from real Tetrix API")
        
        # The extraction_flags endpoint returns the full parsed_document with all financial data!
        try:
            logger.info("Using extraction_flags endpoint to get full parsed document")
            response = await self.session.get(
                f"{self.base_url}/tetrix-analytics-microservice/discrepancies/extraction_flags/{document_path}",
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            if response.status == 200:
                data = await response.json()
                
                # Extract the parsed_document which contains all the real financial data
                parsed_document = data.get("parsed_document", {})
                
                if parsed_document:
                    logger.info(f"SUCCESS! Got full parsed document with {len(parsed_document)} fields")
                    logger.info(f"Fund: {parsed_document.get('fund_name')}")
                    logger.info(f"Assets: {len(parsed_document.get('assets', []))} assets")
                    
                    # Return the parsed document which has all the financial data
                    return parsed_document
                else:
                    logger.warning("No parsed_document found in response")
                    return data
            else:
                logger.warning(f"extraction_flags failed with status {response.status}")
                response_text = await response.text()
                logger.warning(f"Response: {response_text[:200]}...")
                
        except Exception as e:
            logger.warning(f"extraction_flags strategy failed: {e}")
        
        # Fallback to other strategies if extraction_flags fails
        strategies = [
            ("consolidate_document", f"/tetrix-analytics-microservice/cons/consolidate_document/{document_path}"),
            ("normalize_api", f"/tetrix-analytics-microservice/normalizer/normalize_documents_api/client_entity/default")
        ]
        
        for strategy_name, endpoint in strategies:
            try:
                logger.info(f"Trying fallback strategy: {strategy_name}")
                
                if strategy_name == "normalize_api":
                    response = await self.session.post(
                        f"{self.base_url}{endpoint}",
                        json={"doc_path": document_path},
                        timeout=aiohttp.ClientTimeout(total=60)
                    )
                else:
                    response = await self.session.post(
                        f"{self.base_url}{endpoint}",
                        timeout=aiohttp.ClientTimeout(total=60)
                    )
                
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"SUCCESS with fallback strategy {strategy_name}")
                    return data
                    
            except Exception as e:
                logger.warning(f"Fallback strategy {strategy_name} failed: {e}")
                continue
        
        # Complete fallback
        logger.warning("All strategies failed, returning minimal document context")
        return {
            "document_path": document_path,
            "error": "Raw document data not accessible",
            "fallback": True
        }


# Factory function to create appropriate client
def create_analytics_client(use_mock: bool = None) -> TetrixAnalyticsClient:
    """
    Create analytics client - always returns real client
    """
    logger.info("Using real analytics client")
    analytics_url = os.getenv("TETRIX_ANALYTICS_URL", "http://internal-backen-micro-f3m5zasfnrzz-435617696.us-east-2.elb.amazonaws.com")
    return TetrixAnalyticsClient(base_url=analytics_url)

# Export
__all__ = ["TetrixAnalyticsClient", "AnalyticsResponse", "Discrepancy", "FocusPoint", "create_analytics_client"]