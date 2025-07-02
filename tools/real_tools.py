"""
Tools with External API Integration
Tool calling with actual web services and data sources
"""

import asyncio
import aiohttp
import json
import re
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
import sqlite3
import os

logger = logging.getLogger(__name__)

class ToolRegistry:
    """Production tool registry with real external integrations"""
    
    def __init__(self):
        self.tools = {}
        self.usage_metrics = {}
        self.rate_limits = {}
    
    def register_tool(self, func, name: str = None):
        """Register a tool function"""
        tool_name = name or func.__name__
        
        # Generate OpenAI/Anthropic compatible schema
        schema = self._generate_schema(func)
        
        self.tools[tool_name] = {
            "function": func,
            "schema": schema,
            "usage_count": 0,
            "error_count": 0,
            "avg_response_time": 0.0
        }
        
        logger.info(f"Registered real tool: {tool_name}")
    
    def _generate_schema(self, func) -> Dict[str, Any]:
        """Generate tool schema from function signature"""
        import inspect
        
        sig = inspect.signature(func)
        doc = func.__doc__ or "No description"
        
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            if param_name in ['self', 'kwargs']:
                continue
            
            param_type = "string"  # default
            if param.annotation == int:
                param_type = "integer"
            elif param.annotation == float:
                param_type = "number"
            elif param.annotation == bool:
                param_type = "boolean"
            elif param.annotation == list:
                param_type = "array"
            elif param.annotation == dict:
                param_type = "object"
            
            properties[param_name] = {
                "type": param_type,
                "description": f"Parameter {param_name}"
            }
            
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": doc,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool with metrics tracking"""
        if tool_name not in self.tools:
            return {"success": False, "error": f"Tool {tool_name} not found"}
        
        tool_info = self.tools[tool_name]
        start_time = asyncio.get_event_loop().time()
        
        try:
            func = tool_info["function"]
            
            # Execute tool
            if asyncio.iscoroutinefunction(func):
                result = await func(**kwargs)
            else:
                result = func(**kwargs)
            
            # Update metrics
            execution_time = asyncio.get_event_loop().time() - start_time
            tool_info["usage_count"] += 1
            
            # Update average response time
            current_avg = tool_info["avg_response_time"]
            usage_count = tool_info["usage_count"]
            tool_info["avg_response_time"] = (current_avg * (usage_count - 1) + execution_time) / usage_count
            
            logger.info(f"Tool {tool_name} executed successfully in {execution_time:.2f}s")
            
            return {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "tool_name": tool_name
            }
            
        except Exception as e:
            tool_info["error_count"] += 1
            logger.error(f"Tool {tool_name} failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "tool_name": tool_name
            }
    
    def get_tool_schemas(self) -> List[Dict]:
        """Get all tool schemas for LLM function calling"""
        return [tool["schema"] for tool in self.tools.values()]
    
    def get_metrics(self) -> Dict[str, Dict]:
        """Get tool usage metrics"""
        return {
            name: {
                "usage_count": info["usage_count"],
                "error_count": info["error_count"],
                "avg_response_time": info["avg_response_time"],
                "success_rate": (info["usage_count"] - info["error_count"]) / max(info["usage_count"], 1)
            }
            for name, info in self.tools.items()
        }

# Global registry
registry = ToolRegistry()

def tool(name: str = None):
    """Decorator to register real tools"""
    def decorator(func):
        registry.register_tool(func, name)
        return func
    return decorator

# Real Financial Data Tools
@tool("yahoo_finance_lookup")
async def yahoo_finance_lookup(symbol: str, period: str = "1y") -> Dict[str, Any]:
    """Get real financial data from Yahoo Finance API"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Get historical data
        hist = ticker.history(period=period)
        
        # Get company info
        info = ticker.info
        
        # Calculate recent performance
        recent_prices = hist['Close'].tail(5).tolist()
        current_price = recent_prices[-1] if recent_prices else None
        
        return {
            "symbol": symbol,
            "current_price": current_price,
            "company_name": info.get("longName", "Unknown"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "recent_prices": recent_prices,
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "data_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": f"Failed to fetch data for {symbol}: {str(e)}"}

@tool("sec_company_search")
async def sec_company_search(company_name: str) -> Dict[str, Any]:
    """Search SEC EDGAR database for real company filings"""
    try:
        # Real SEC API call
        base_url = "https://www.sec.gov/cgi-bin/browse-edgar"
        
        session = aiohttp.ClientSession(
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )
        
        params = {
            "action": "getcompany",
            "company": company_name,
            "type": "10-K",
            "dateb": "",
            "count": "5",
            "output": "xml"
        }
        
        async with session.get(base_url, params=params) as response:
            if response.status == 200:
                text = await response.text()
                
                # Parse XML response (simplified)
                # In production, you'd use proper XML parsing
                if "No matching companies" in text:
                    return {"found": False, "message": "No matching companies found"}
                
                return {
                    "found": True,
                    "company_name": company_name,
                    "search_timestamp": datetime.now().isoformat(),
                    "filings_available": True,
                    "sec_url": f"{base_url}?{aiohttp.helpers.urlencode(params)}"
                }
            else:
                return {"error": f"SEC API returned status {response.status}"}
                
        await session.close()
        
    except Exception as e:
        return {"error": f"SEC search failed: {str(e)}"}

@tool("real_time_calculator")
async def real_time_calculator(expression: str) -> Dict[str, Any]:
    """Perform real mathematical calculations with safety checks"""
    try:
        # Sanitize expression for safety
        allowed_chars = set("0123456789+-*/().% ")
        if not all(c in allowed_chars for c in expression):
            return {"error": "Expression contains invalid characters"}
        
        # Additional safety checks
        if any(dangerous in expression.lower() for dangerous in ["import", "exec", "eval", "__"]):
            return {"error": "Expression contains dangerous operations"}
        
        # Evaluate safely
        result = eval(expression)
        
        return {
            "expression": expression,
            "result": result,
            "calculation_time": datetime.now().isoformat(),
            "type": type(result).__name__
        }
        
    except ZeroDivisionError:
        return {"error": "Division by zero"}
    except Exception as e:
        return {"error": f"Calculation failed: {str(e)}"}

@tool("web_search")
async def web_search(query: str, num_results: int = 5) -> Dict[str, Any]:
    """Perform real web search using DuckDuckGo API"""
    try:
        # Use DuckDuckGo Instant Answer API (no API key required)
        session = aiohttp.ClientSession()
        
        # Search API
        search_url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1"
        }
        
        async with session.get(search_url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                
                # Extract results
                results = []
                
                # Abstract result
                if data.get("Abstract"):
                    results.append({
                        "title": data.get("Heading", ""),
                        "snippet": data["Abstract"],
                        "url": data.get("AbstractURL", ""),
                        "source": data.get("AbstractSource", "")
                    })
                
                # Related topics
                for topic in data.get("RelatedTopics", [])[:num_results-len(results)]:
                    if isinstance(topic, dict) and "Text" in topic:
                        results.append({
                            "title": topic.get("FirstURL", "").split("/")[-1].replace("_", " "),
                            "snippet": topic["Text"],
                            "url": topic.get("FirstURL", ""),
                            "source": "DuckDuckGo"
                        })
                
                await session.close()
                
                return {
                    "query": query,
                    "results": results,
                    "num_results": len(results),
                    "search_timestamp": datetime.now().isoformat()
                }
            else:
                await session.close()
                return {"error": f"Search API returned status {response.status}"}
                
    except Exception as e:
        return {"error": f"Web search failed: {str(e)}"}

@tool("database_query")
async def database_query(query_type: str, entity_name: str = None) -> Dict[str, Any]:
    """Query real financial database (SQLite for demo, but shows real DB integration)"""
    try:
        # Create/connect to real database
        db_path = "financial_data.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS companies (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                sector TEXT,
                market_cap REAL,
                last_updated TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fund_data (
                id INTEGER PRIMARY KEY,
                fund_name TEXT,
                aum REAL,
                inception_date DATE,
                strategy TEXT,
                last_updated TIMESTAMP
            )
        """)
        
        # Insert sample data if empty
        cursor.execute("SELECT COUNT(*) FROM companies")
        if cursor.fetchone()[0] == 0:
            sample_companies = [
                ("Apple Inc", "Technology", 3000000000000, datetime.now()),
                ("Blackstone Inc", "Financial Services", 950000000000, datetime.now()),
                ("Microsoft Corporation", "Technology", 2800000000000, datetime.now())
            ]
            cursor.executemany(
                "INSERT INTO companies (name, sector, market_cap, last_updated) VALUES (?, ?, ?, ?)",
                sample_companies
            )
            
            sample_funds = [
                ("Blackstone Capital Partners VII", 24500000000, "2017-01-01", "Buyout", datetime.now()),
                ("Apollo Global Management Fund IX", 18000000000, "2019-03-01", "Buyout", datetime.now()),
                ("KKR Americas Fund XIII", 15000000000, "2020-06-01", "Buyout", datetime.now())
            ]
            cursor.executemany(
                "INSERT INTO fund_data (fund_name, aum, inception_date, strategy, last_updated) VALUES (?, ?, ?, ?, ?)",
                sample_funds
            )
            
            conn.commit()
        
        # Execute query based on type
        if query_type == "company_lookup" and entity_name:
            cursor.execute(
                "SELECT * FROM companies WHERE name LIKE ? OR name LIKE ?",
                (f"%{entity_name}%", f"%{entity_name.split()[0]}%")
            )
            
            results = cursor.fetchall()
            if results:
                company = results[0]
                return {
                    "found": True,
                    "company_name": company[1],
                    "sector": company[2],
                    "market_cap": company[3],
                    "last_updated": company[4]
                }
            else:
                return {"found": False, "message": f"No company found matching {entity_name}"}
        
        elif query_type == "fund_lookup" and entity_name:
            cursor.execute(
                "SELECT * FROM fund_data WHERE fund_name LIKE ?",
                (f"%{entity_name}%",)
            )
            
            results = cursor.fetchall()
            if results:
                fund = results[0]
                return {
                    "found": True,
                    "fund_name": fund[1],
                    "aum": fund[2],
                    "inception_date": fund[3],
                    "strategy": fund[4],
                    "last_updated": fund[5]
                }
            else:
                return {"found": False, "message": f"No fund found matching {entity_name}"}
        
        else:
            return {"error": "Invalid query type or missing entity name"}
            
        conn.close()
        
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@tool("exchange_rate_lookup")
async def exchange_rate_lookup(from_currency: str, to_currency: str) -> Dict[str, Any]:
    """Get real-time exchange rates from external API"""
    try:
        # Use free exchange rate API
        session = aiohttp.ClientSession()
        
        # Free API that doesn't require key
        url = f"https://api.exchangerate-api.com/v4/latest/{from_currency.upper()}"
        
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                
                to_currency_upper = to_currency.upper()
                if to_currency_upper in data.get("rates", {}):
                    rate = data["rates"][to_currency_upper]
                    
                    await session.close()
                    
                    return {
                        "from_currency": from_currency.upper(),
                        "to_currency": to_currency_upper,
                        "exchange_rate": rate,
                        "last_updated": data.get("date"),
                        "base_currency": data.get("base"),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    await session.close()
                    return {"error": f"Currency {to_currency} not found"}
            else:
                await session.close()
                return {"error": f"Exchange rate API returned status {response.status}"}
                
    except Exception as e:
        return {"error": f"Exchange rate lookup failed: {str(e)}"}

@tool("date_business_logic")
async def date_business_logic(date_string: str, operation: str = "validate") -> Dict[str, Any]:
    """Real date validation with business logic and timezone support"""
    try:
        # Parse date with multiple format support
        parsed_date = pd.to_datetime(date_string, utc=True)
        
        if operation == "validate":
            # Business day logic
            is_business_day = parsed_date.weekday() < 5
            
            # Holiday checking (simplified - in production use holidays library)
            current_year = parsed_date.year
            major_holidays = [
                f"{current_year}-01-01",  # New Year
                f"{current_year}-07-04",  # Independence Day
                f"{current_year}-12-25"   # Christmas
            ]
            
            is_holiday = date_string in major_holidays
            
            # Financial quarter logic
            quarter = (parsed_date.month - 1) // 3 + 1
            quarter_end = parsed_date.month % 3 == 0 and parsed_date.day >= 28
            
            # Time zone considerations
            eastern = parsed_date.tz_convert('US/Eastern')
            
            return {
                "is_valid": True,
                "original_string": date_string,
                "parsed_date": parsed_date.isoformat(),
                "is_business_day": is_business_day,
                "is_holiday": is_holiday,
                "quarter": quarter,
                "is_quarter_end": quarter_end,
                "day_of_week": parsed_date.strftime("%A"),
                "eastern_time": eastern.isoformat(),
                "days_from_now": (parsed_date - pd.Timestamp.now(tz='UTC')).days
            }
            
        elif operation == "business_days_between":
            # This would need a second date parameter in real implementation
            return {"error": "business_days_between requires two dates"}
            
        else:
            return {"error": f"Unknown operation: {operation}"}
            
    except Exception as e:
        return {"error": f"Date processing failed: {str(e)}"}

@tool("text_extraction")
async def text_extraction(text: str, extraction_type: str = "numbers") -> Dict[str, Any]:
    """Extract structured information from text using real NLP techniques"""
    try:
        if extraction_type == "numbers":
            # Extract various number formats
            patterns = {
                "integers": r'\b\d+\b',
                "decimals": r'\b\d+\.\d+\b',
                "currency": r'\$[\d,]+\.?\d*',
                "percentages": r'\d+\.?\d*%',
                "millions": r'\d+\.?\d*\s*million',
                "billions": r'\d+\.?\d*\s*billion'
            }
            
            extracted = {}
            for pattern_name, pattern in patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                extracted[pattern_name] = matches
            
            # Convert text numbers
            text_numbers = {
                "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "million": 1000000, "billion": 1000000000
            }
            
            converted_numbers = []
            words = text.lower().split()
            for i, word in enumerate(words):
                if word in text_numbers:
                    if i + 1 < len(words) and words[i + 1] in ["million", "billion"]:
                        multiplier = text_numbers[words[i + 1]]
                        converted_numbers.append(text_numbers[word] * multiplier)
                    else:
                        converted_numbers.append(text_numbers[word])
            
            return {
                "text": text,
                "extraction_type": extraction_type,
                "patterns_found": extracted,
                "converted_numbers": converted_numbers,
                "extraction_timestamp": datetime.now().isoformat()
            }
            
        elif extraction_type == "entities":
            # Simple entity extraction
            # In production, use spaCy or similar NLP library
            
            # Find potential company names (capitalized words)
            company_patterns = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Inc|LLC|Corp|Ltd|Fund|Capital|Partners|Group))\b'
            companies = re.findall(company_patterns, text)
            
            # Find dates
            date_patterns = r'\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{4}\b'
            dates = re.findall(date_patterns, text)
            
            return {
                "text": text,
                "companies": companies,
                "dates": dates,
                "extraction_timestamp": datetime.now().isoformat()
            }
            
        else:
            return {"error": f"Unknown extraction type: {extraction_type}"}
            
    except Exception as e:
        return {"error": f"Text extraction failed: {str(e)}"}

# Tool metrics and management
async def get_tool_health() -> Dict[str, Any]:
    """Get comprehensive tool health metrics"""
    metrics = registry.get_metrics()
    
    total_calls = sum(tool["usage_count"] for tool in metrics.values())
    total_errors = sum(tool["error_count"] for tool in metrics.values())
    
    return {
        "total_tools": len(metrics),
        "total_calls": total_calls,
        "total_errors": total_errors,
        "overall_success_rate": (total_calls - total_errors) / max(total_calls, 1),
        "tool_metrics": metrics,
        "health_check_timestamp": datetime.now().isoformat()
    }

# Export for use in agents
__all__ = ["registry", "tool", "get_tool_health"]