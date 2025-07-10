#!/usr/bin/env python3
"""
Test script to demonstrate the improvement score issue and validate the fix
"""

import asyncio
import json
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ImprovementScoreTester:
    """Test the improvement score calculation and document matching"""
    
    def __init__(self):
        self.test_cases = [
            {
                "name": "ABRY Partners VIII, L.P.",
                "corrected_doc": {
                    "fund_name": "ABRY Partners VIII, L.P.",
                    "fund_org_id": "None",
                    "reporting_date": "2024-12-31"
                },
                "consolidated_docs": [
                    {
                        "fund_name": "ABRY Partners VIII",
                        "fund_org_id": "12345",
                        "reporting_date": "2024-12-31 00:00:00"
                    },
                    {
                        "fund_name": "ABRY Partners Fund VIII, L.P.",
                        "fund_org_id": "67890",
                        "reporting_date": "2024-12-31 00:00:00"
                    },
                    {
                        "fund_name": "Different Fund",
                        "fund_org_id": "11111",
                        "reporting_date": "2024-12-31 00:00:00"
                    }
                ]
            },
            {
                "name": "Crestview Partners III, L.P.",
                "corrected_doc": {
                    "fund_name": "Crestview Partners III, L.P.",
                    "fund_org_id": "None",
                    "reporting_date": "2024-12-31"
                },
                "consolidated_docs": [
                    {
                        "fund_name": "Crestview Partners III",
                        "fund_org_id": "22222",
                        "reporting_date": "2024-12-31 00:00:00"
                    }
                ]
            }
        ]
    
    def test_normalize_fund_name(self):
        """Test the fund name normalization function"""
        print("🧪 Testing Fund Name Normalization")
        print("=" * 50)
        
        test_names = [
            "ABRY Partners VIII, L.P.",
            "ABRY Partners Fund VIII, L.P.",
            "Crestview Partners III, L.P.",
            "Crestview Partners III",
            "Fund I, LP",
            "Fund II Limited Partnership"
        ]
        
        for name in test_names:
            normalized = self._normalize_fund_name(name)
            print(f"'{name}' → '{normalized}'")
        
        print()
    
    def _normalize_fund_name(self, name: str) -> str:
        """Normalize fund name for better matching (same as in document_agent.py)"""
        if not name:
            return ""
        
        # Convert to lowercase and remove extra whitespace
        normalized = name.lower().strip()
        
        # Remove common suffixes and variations
        suffixes_to_remove = [
            ', l.p.', ' l.p.', ' lp', ' llc', ' inc', ' ltd', ' limited',
            ', lp', ' lp.', ' l.p', ' limited partnership',
            ' fund', ' fund i', ' fund ii', ' fund iii', ' fund iv', ' fund v',
            ' fund vi', ' fund vii', ' fund viii', ' fund ix', ' fund x',
            ' i', ' ii', ' iii', ' iv', ' v', ' vi', ' vii', ' viii', ' ix', ' x'
        ]
        
        for suffix in suffixes_to_remove:
            normalized = normalized.replace(suffix, '')
        
        # Remove punctuation and extra spaces
        normalized = normalized.replace(',', '').replace('.', '').replace('  ', ' ').strip()
        
        return normalized
    
    def test_document_matching(self):
        """Test the improved document matching logic"""
        print("🧪 Testing Document Matching Logic")
        print("=" * 50)
        
        for test_case in self.test_cases:
            print(f"\n📋 Test Case: {test_case['name']}")
            print(f"   Corrected Doc: {test_case['corrected_doc']['fund_name']}")
            
            matches_found = 0
            for i, consolidated_doc in enumerate(test_case['consolidated_docs']):
                if self._documents_match(test_case['corrected_doc'], consolidated_doc):
                    print(f"    MATCH {i+1}: {consolidated_doc['fund_name']}")
                    matches_found += 1
                else:
                    print(f"   ❌ NO MATCH {i+1}: {consolidated_doc['fund_name']}")
            
            print(f"   Total matches: {matches_found}")
        
        print()
    
    def _documents_match(self, corrected_doc: Dict[str, Any], consolidated_doc: Dict[str, Any]) -> bool:
        """Improved document matching logic (same as in document_agent.py)"""
        # Match by fund organization ID and reporting date (if both available)
        corrected_org_id = corrected_doc.get("fund_org_id")
        consolidated_org_id = consolidated_doc.get("fund_org_id")
        
        if (corrected_org_id and consolidated_org_id and 
            corrected_org_id != "None" and consolidated_org_id != "None" and
            corrected_org_id == consolidated_org_id and
            corrected_doc.get("reporting_date") == consolidated_doc.get("reporting_date")):
            return True
        
        # Enhanced fund name matching with better normalization
        corrected_name = corrected_doc.get("fund_name", "").strip()
        consolidated_name = consolidated_doc.get("fund_name", "").strip()
        
        if corrected_name and consolidated_name:
            # Normalize both names for comparison
            corrected_normalized = self._normalize_fund_name(corrected_name)
            consolidated_normalized = self._normalize_fund_name(consolidated_name)
            
            # Exact match after normalization
            if corrected_normalized == consolidated_normalized:
                return True
            
            # Partial match (one name contains the other)
            if (corrected_normalized in consolidated_normalized or 
                consolidated_normalized in corrected_normalized):
                return True
            
            # Try matching by key words (e.g., "ABRY Partners" should match "ABRY Partners VIII")
            corrected_words = set(corrected_normalized.split())
            consolidated_words = set(consolidated_normalized.split())
            
            # If more than 50% of words match, consider it a match
            common_words = corrected_words & consolidated_words
            if (len(common_words) >= min(len(corrected_words), len(consolidated_words)) * 0.5 and
                len(common_words) >= 2):  # At least 2 words must match
                return True
        
        return False
    
    def analyze_improvement_score_issue(self):
        """Analyze why improvement scores are returning 0"""
        print(" Analysis: Why Improvement Scores Return 0")
        print("=" * 60)
        
        print("""
ROOT CAUSE ANALYSIS:

1. **Document Matching Failure**
   - The system tries to match corrected documents with consolidated documents
   - When no matches are found, no validation can occur
   - This results in 0 successful measurements

2. **Fund Name Variations**
   - "ABRY Partners VIII, L.P." vs "ABRY Partners VIII"
   - "Crestview Partners III, L.P." vs "Crestview Partners III"
   - Missing fund_org_id values ("None")

3. **Improvement Calculation Logic**
   - When no successful measurements exist, _calculate_overall_improvement returns:
     - average_improvement_percentage: 0.0
     - documents_improved: 0
     - improvement_rate: 0.0

4. **Why Your Evaluation Script Works**
   - Uses direct field-by-field comparison
   - Better normalization of fund names
   - Focuses on accuracy rather than "improvement"
   - Handles missing fund_org_id gracefully

SOLUTION:
- Improved document matching with better fund name normalization
- Enhanced matching logic for partial matches and word similarity
- Better handling of missing fund_org_id values
        """)
    
    def demonstrate_fix(self):
        """Demonstrate how the fix improves matching"""
        print("🔧 Demonstrating the Fix")
        print("=" * 40)
        
        print("""
BEFORE FIX:
- Only exact fund name matches
- No handling of missing fund_org_id
- No partial matching

AFTER FIX:
- Normalized fund name matching
- Partial name matching
- Word similarity matching
- Better handling of missing IDs

This should result in:
- More successful document matches
- Actual improvement measurements
- Non-zero improvement scores
        """)

def main():
    """Run the improvement score analysis"""
    print("🔬 IMPROVEMENT SCORE ANALYSIS & FIX")
    print("=" * 80)
    
    tester = ImprovementScoreTester()
    
    # Run tests
    tester.test_normalize_fund_name()
    tester.test_document_matching()
    tester.analyze_improvement_score_issue()
    tester.demonstrate_fix()
    
    print("\n Analysis Complete!")
    print("\nNext Steps:")
    print("1. The improved document matching logic has been applied to document_agent.py")
    print("2. Test the system again to see if improvement scores are now non-zero")
    print("3. Monitor the logs for successful document matches")

if __name__ == "__main__":
    main() 