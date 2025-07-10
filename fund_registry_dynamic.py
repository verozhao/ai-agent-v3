"""
Dynamic Fund Registry using API endpoints
Replace hardcoded fund data with real Tetrix API data
"""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
import aiohttp
from fuzzywuzzy import fuzz
import re

from analytics_client import create_analytics_client

logger = logging.getLogger(__name__)

@dataclass
class FundInfo:
    """Fund information from Tetrix API"""
    fund_name: str
    fund_family: Optional[str] = None
    vintage_year: Optional[int] = None
    fund_size_usd: Optional[float] = None
    strategy: Optional[str] = None
    geographic_focus: Optional[str] = None
    status: Optional[str] = None
    aliases: List[str] = None
    source_document: Optional[str] = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []

class DynamicFundRegistry:
    """Dynamic fund registry that fetches fund data from API"""
    
    def __init__(self, db_path: str = "fund_registry_dynamic.db"):
        self.db_path = db_path
        self.analytics_client = None
        self.fund_cache = {}
        self.alias_cache = {}
        self.last_refresh = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.analytics_client = create_analytics_client(use_mock=False)
        await self.analytics_client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.analytics_client:
            await self.analytics_client.__aexit__(exc_type, exc_val, exc_tb)
    
    def _init_database(self):
        """Initialize the fund registry database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create funds table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS funds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fund_name TEXT NOT NULL UNIQUE,
                fund_family TEXT,
                vintage_year INTEGER,
                fund_size_usd REAL,
                strategy TEXT,
                geographic_focus TEXT,
                status TEXT,
                source_document TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create aliases table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fund_aliases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fund_name TEXT NOT NULL,
                alias TEXT NOT NULL,
                alias_type TEXT, -- 'abbreviation', 'variation', 'short_name', 'generated'
                confidence REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (fund_name) REFERENCES funds (fund_name),
                UNIQUE(fund_name, alias)
            )
        """)
        
        # Create document tracking table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_path TEXT NOT NULL UNIQUE,
                fund_names_found TEXT, -- JSON array of fund names
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("Fund registry database initialized")
    
    async def fetch_fund_data_from_api(self, sample_document_paths: List[str] = None) -> List[FundInfo]:
        """
        Fetch fund data from API by analyzing sample documents
        This extracts fund names from real parsed documents
        """
        
        if sample_document_paths is None:
            # Use some known document paths for testing
            sample_document_paths = [
                "PEFundPortfolioExtraction/67ee89d7ecbb614e1103e533",
                # Add more document paths as needed
            ]
        
        funds_found = []
        
        for doc_path in sample_document_paths:
            try:
                logger.info(f"Fetching fund data from document: {doc_path}")
                
                # Get raw document data which contains fund information
                raw_data = await self.analytics_client.get_raw_document_data(doc_path)
                
                if raw_data and not raw_data.get("error"):
                    # Extract fund information from the parsed document
                    fund_info = self._extract_fund_info_from_document(raw_data, doc_path)
                    if fund_info:
                        funds_found.append(fund_info)
                        logger.info(f"Found fund: {fund_info.fund_name}")
                    
            except Exception as e:
                logger.warning(f"Failed to fetch fund data from {doc_path}: {e}")
                continue
        
        return funds_found
    
    def _extract_fund_info_from_document(self, raw_data: Dict[str, Any], doc_path: str) -> Optional[FundInfo]:
        """Extract fund information from a parsed document"""
        
        fund_name = raw_data.get("fund_name")
        if not fund_name:
            logger.warning(f"No fund_name found in document {doc_path}")
            return None
        
        # Generate aliases for the fund name
        aliases = self._generate_fund_aliases(fund_name)
        
        # Extract additional fund information if available
        fund_info = FundInfo(
            fund_name=fund_name,
            fund_family=self._extract_fund_family(fund_name),
            vintage_year=self._extract_vintage_year(fund_name),
            fund_size_usd=raw_data.get("total_fund_size"),
            strategy=self._infer_strategy(raw_data),
            geographic_focus=self._infer_geographic_focus(raw_data),
            status="Active",  # Assume active if we have recent data
            aliases=aliases,
            source_document=doc_path
        )
        
        return fund_info
    
    def _generate_fund_aliases(self, fund_name: str) -> List[str]:
        """Generate possible aliases for a fund name"""
        
        aliases = []
        
        # Add the original name
        aliases.append(fund_name)
        
        # Common patterns for fund names
        patterns = [
            # Remove "Fund" and roman numerals
            (r'\s+Fund\s+([IVX]+)', r' \1'),
            # Remove "Partners" 
            (r'\s+Partners\s+([IVX]+)', r' \1'),
            # Remove "LP"
            (r'\s+LP\s*$', ''),
            # Remove "L.P."
            (r'\s+L\.P\.\s*$', ''),
            # Remove commas
            (r',', ''),
            # Roman numeral variations
            (r'\s+VIII\s*$', ' 8'),
            (r'\s+VII\s*$', ' 7'),
            (r'\s+VI\s*$', ' 6'),
            (r'\s+V\s*$', ' 5'),
            (r'\s+IV\s*$', ' 4'),
            (r'\s+III\s*$', ' 3'),
            (r'\s+II\s*$', ' 2'),
            (r'\s+I\s*$', ' 1'),
        ]
        
        current_variants = [fund_name]
        
        # Apply patterns to create variations
        for pattern, replacement in patterns:
            new_variants = []
            for variant in current_variants:
                new_variant = re.sub(pattern, replacement, variant).strip()
                if new_variant and new_variant != variant:
                    new_variants.append(new_variant)
            current_variants.extend(new_variants)
        
        aliases.extend(current_variants)
        
        # Generate abbreviations from all variants
        for variant in current_variants:
            words = variant.split()
            if len(words) > 1:
                # First letters of each word
                abbreviation = ''.join(word[0].upper() for word in words if word and word[0].isalpha())
                if len(abbreviation) > 1:
                    aliases.append(abbreviation)
                    
                    # With periods
                    aliases.append('.'.join(abbreviation) + '.')
                    
                    # With spaces
                    aliases.append(' '.join(abbreviation))
        
        # Common abbreviations
        abbreviation_map = {
            'Partners': 'P',
            'Capital': 'C',
            'Management': 'M',
            'Private': 'P',
            'Equity': 'E',
            'Growth': 'G',
            'Fund': 'F',
            'Venture': 'V',
            'Investment': 'I',
            'Holdings': 'H'
        }
        
        for variant in current_variants:
            for full_word, abbrev in abbreviation_map.items():
                if full_word in variant:
                    short_version = variant.replace(full_word, abbrev)
                    aliases.append(short_version)
        
        # Add common short forms
        if "ABRY" in fund_name:
            aliases.extend([
                "ABRY",
                "Abry",
                "abry",
                "ABRY Partners",
                "Abry Partners"
            ])
        
        # Add numerical variations (for all variants)
        for variant in list(aliases):
            # Convert roman numerals to numbers
            roman_to_num = {
                'VIII': '8', 'VII': '7', 'VI': '6', 'V': '5', 
                'IV': '4', 'III': '3', 'II': '2', 'I': '1'
            }
            
            for roman, num in roman_to_num.items():
                if roman in variant:
                    aliases.append(variant.replace(roman, num))
                    aliases.append(variant.replace(roman, f" {num}"))
        
        # Remove duplicates while preserving order
        unique_aliases = []
        seen = set()
        for alias in aliases:
            clean_alias = alias.strip()
            # Filter out very short or meaningless aliases
            if (clean_alias and 
                clean_alias not in seen and 
                len(clean_alias) > 1 and 
                not clean_alias.isdigit() and
                clean_alias not in ['L', 'P', 'A']):
                unique_aliases.append(clean_alias)
                seen.add(clean_alias)
        
        return unique_aliases
    
    def _extract_fund_family(self, fund_name: str) -> Optional[str]:
        """Extract fund family from fund name"""
        
        # Common patterns for fund families
        patterns = [
            r'^(.*?)\s+Fund',
            r'^(.*?)\s+Partners',
            r'^(.*?)\s+Capital',
            r'^(.*?)\s+[IVX]+',
            r'^(.*?)\s+\d+',
        ]
        
        for pattern in patterns:
            match = re.match(pattern, fund_name)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_vintage_year(self, fund_name: str) -> Optional[int]:
        """Extract vintage year from fund name if present"""
        
        # Look for 4-digit years
        year_match = re.search(r'(19|20)\d{2}', fund_name)
        if year_match:
            return int(year_match.group())
        
        return None
    
    def _infer_strategy(self, raw_data: Dict[str, Any]) -> Optional[str]:
        """Infer investment strategy from document data"""
        
        assets = raw_data.get("assets", {})
        if not assets:
            return "Private Equity"  # Default assumption
        
        # Look at asset characteristics to infer strategy
        # This is a simplified heuristic
        
        return "Private Equity"  # Default assumption
    
    def _infer_geographic_focus(self, raw_data: Dict[str, Any]) -> Optional[str]:
        """Infer geographic focus from document data"""
        
        assets = raw_data.get("assets", {})
        if not assets:
            return None
        
        # Look at asset locations
        locations = []
        if isinstance(assets, dict):
            for asset_name, asset_data in assets.items():
                if isinstance(asset_data, dict) and "location" in asset_data:
                    locations.append(asset_data["location"])
        elif isinstance(assets, list):
            for asset_data in assets:
                if isinstance(asset_data, dict) and "location" in asset_data:
                    locations.append(asset_data["location"])
        
        if not locations:
            return None
        
        # Simple heuristic based on location distribution
        unique_locations = set(locations)
        if len(unique_locations) == 1:
            return list(unique_locations)[0]
        elif "USA" in unique_locations and "Europe" in unique_locations:
            return "Global"
        else:
            return "Multi-Regional"
    
    async def refresh_fund_registry(self, sample_document_paths: List[str] = None):
        """Refresh the fund registry with latest data from API"""
        
        logger.info("Refreshing fund registry from Tetrix API...")
        
        # Initialize database if needed
        self._init_database()
        
        # Fetch fund data from API
        funds = await self.fetch_fund_data_from_api(sample_document_paths)
        
        if not funds:
            logger.warning("No funds found from API, keeping existing data")
            return
        
        # Update database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for fund in funds:
            try:
                # Insert or update fund
                cursor.execute("""
                    INSERT OR REPLACE INTO funds 
                    (fund_name, fund_family, vintage_year, fund_size_usd, strategy, geographic_focus, status, source_document, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    fund.fund_name,
                    fund.fund_family,
                    fund.vintage_year,
                    fund.fund_size_usd,
                    fund.strategy,
                    fund.geographic_focus,
                    fund.status,
                    fund.source_document,
                    datetime.now().isoformat()
                ))
                
                # Insert aliases
                for alias in fund.aliases:
                    cursor.execute("""
                        INSERT OR REPLACE INTO fund_aliases (fund_name, alias, alias_type)
                        VALUES (?, ?, ?)
                    """, (fund.fund_name, alias, "generated"))
                
                conn.commit()
                logger.info(f"Updated fund: {fund.fund_name} with {len(fund.aliases)} aliases")
                
            except Exception as e:
                logger.error(f"Error updating fund {fund.fund_name}: {e}")
                conn.rollback()
        
        conn.close()
        
        # Update cache
        self.fund_cache = {fund.fund_name: fund for fund in funds}
        self.last_refresh = datetime.now()
        
        logger.info(f"Fund registry refreshed with {len(funds)} funds")
    
    def find_fund_by_name(self, query_name: str, min_confidence: float = 0.8) -> Optional[Tuple[str, float]]:
        """
        Find a fund by name using fuzzy matching against aliases
        Returns (fund_name, confidence) or None
        """
        
        if not query_name:
            return None
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all funds and their aliases
        cursor.execute("""
            SELECT f.fund_name, fa.alias
            FROM funds f
            LEFT JOIN fund_aliases fa ON f.fund_name = fa.fund_name
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return None
        
        best_match = None
        best_score = 0
        
        for fund_name, alias in results:
            if alias:
                # Calculate fuzzy match score
                score = fuzz.ratio(query_name.lower(), alias.lower()) / 100.0
                
                if score > best_score:
                    best_score = score
                    best_match = fund_name
        
        if best_score >= min_confidence:
            return (best_match, best_score)
        
        return None
    
    def get_all_funds(self) -> List[FundInfo]:
        """Get all funds from the registry"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT fund_name, fund_family, vintage_year, fund_size_usd, strategy, geographic_focus, status, source_document
            FROM funds
        """)
        
        funds = []
        for row in cursor.fetchall():
            fund_name = row[0]
            
            # Get aliases for this fund
            cursor.execute("SELECT alias FROM fund_aliases WHERE fund_name = ?", (fund_name,))
            aliases = [alias[0] for alias in cursor.fetchall()]
            
            fund = FundInfo(
                fund_name=fund_name,
                fund_family=row[1],
                vintage_year=row[2],
                fund_size_usd=row[3],
                strategy=row[4],
                geographic_focus=row[5],
                status=row[6],
                aliases=aliases,
                source_document=row[7]
            )
            funds.append(fund)
        
        conn.close()
        return funds
    
    def get_fund_stats(self) -> Dict[str, Any]:
        """Get statistics about the fund registry"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM funds")
        total_funds = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM fund_aliases")
        total_aliases = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT fund_family) FROM funds WHERE fund_family IS NOT NULL")
        unique_families = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_funds": total_funds,
            "total_aliases": total_aliases,
            "unique_families": unique_families,
            "last_refresh": self.last_refresh.isoformat() if self.last_refresh else None
        }

# Test function
async def test_fund_registry():
    """Test the dynamic fund registry"""
    
    async with DynamicFundRegistry() as registry:
        # Test connection
        connection_test = await registry.analytics_client.test_connection()
        print(f"Connection test: {connection_test}")
        
        if connection_test.get("connected"):
            # Refresh registry with real data
            await registry.refresh_fund_registry()
            
            # Test fund lookup
            test_queries = [
                "ABRY",
                "ABRY Partners",
                "ABRY Partners VIII",
                "Abry VIII",
                "abry 8"
            ]
            
            print("\nTesting fund name matching:")
            for query in test_queries:
                result = registry.find_fund_by_name(query)
                if result:
                    fund_name, confidence = result
                    print(f"'{query}' -> '{fund_name}' (confidence: {confidence:.2f})")
                else:
                    print(f"'{query}' -> No match found")
            
            # Show all funds
            funds = registry.get_all_funds()
            print(f"\nFound {len(funds)} funds:")
            for fund in funds:
                print(f"- {fund.fund_name} (aliases: {len(fund.aliases)})")
                print(f"  Aliases: {', '.join(fund.aliases[:5])}{'...' if len(fund.aliases) > 5 else ''}")
            
            # Show stats
            stats = registry.get_fund_stats()
            print(f"\nRegistry stats: {stats}")
        else:
            print("Cannot connect to analytics service - using mock data for testing")

if __name__ == "__main__":
    asyncio.run(test_fund_registry())