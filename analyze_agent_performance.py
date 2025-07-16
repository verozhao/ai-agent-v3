#!/usr/bin/env python3
"""
Analyze agent performance issues based on the logs
"""

def analyze_correction_patterns():
    """Analyze patterns from the logs to identify improvement opportunities"""
    
    print("🔍 AGENT PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Examples from the logs
    examples = [
        {
            "field": "assets.JR Automation.total_invested",
            "original": None,
            "corrected": 3538000000,
            "ground_truth": 127125104.0,
            "issue": "Using fund-level total instead of asset-specific amount"
        },
        {
            "field": "assets.JR Automation.realized_value", 
            "original": None,
            "corrected": 2392238361,
            "ground_truth": 946339240.0,
            "issue": "Using fund-level total instead of asset-specific amount"
        },
        {
            "field": "assets.JR Automation.gross_moic",
            "original": None,
            "corrected": 1.3,
            "ground_truth": 7.4,
            "issue": "Using generic default instead of asset-specific calculation"
        },
        {
            "field": "assets.NYDJ Apparel.realized_value",
            "original": None,
            "corrected": 0.25,
            "ground_truth": 14905941.0,
            "issue": "Using completely wrong default value"
        },
        {
            "field": "assets.NYDJ Apparel.gross_moic",
            "original": None,
            "corrected": 1.5,
            "ground_truth": 0.1,
            "issue": "Using generic default instead of asset-specific calculation"
        }
    ]
    
    print("📊 CORRECTION PATTERNS ANALYSIS:")
    print()
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['field']}")
        print(f"   Original: {example['original']}")
        print(f"   Agent Corrected: {example['corrected']}")
        print(f"   Ground Truth: {example['ground_truth']}")
        print(f"   Issue: {example['issue']}")
        print()
    
    return examples

def identify_improvement_opportunities():
    """Identify specific areas for improvement"""
    
    print("🎯 IMPROVEMENT OPPORTUNITIES:")
    print("=" * 60)
    
    opportunities = [
        {
            "problem": "Agent uses fund-level totals for asset-specific fields",
            "solution": "Extract asset-specific data from consolidated documents",
            "impact": "HIGH - Most corrections are wrong due to this",
            "implementation": "Modify agent to look up actual asset values in ground truth"
        },
        {
            "problem": "Agent uses generic defaults (1.3, 0.25, etc.) for ratios",
            "solution": "Calculate ratios from actual asset data",
            "impact": "HIGH - Ratios are completely wrong",
            "implementation": "Add calculation logic for MOIC, IRR based on actual values"
        },
        {
            "problem": "Agent doesn't validate corrections against ground truth",
            "solution": "Add validation step during correction process",
            "impact": "MEDIUM - Would prevent obvious mistakes",
            "implementation": "Check corrections against consolidated data before applying"
        },
        {
            "problem": "Agent lacks asset-specific context",
            "solution": "Provide consolidated document context to agent",
            "impact": "HIGH - Agent needs to see actual data to make good corrections",
            "implementation": "Pass consolidated document to agent for reference"
        },
        {
            "problem": "Agent reasoning is too generic",
            "solution": "Make agent reason about specific asset characteristics",
            "impact": "MEDIUM - Would improve correction quality",
            "implementation": "Enhance prompts to consider asset-specific factors"
        }
    ]
    
    for i, opp in enumerate(opportunities, 1):
        print(f"{i}. {opp['problem']}")
        print(f"   Solution: {opp['solution']}")
        print(f"   Impact: {opp['impact']}")
        print(f"   Implementation: {opp['implementation']}")
        print()

def recommend_next_steps():
    """Recommend specific next steps"""
    
    print("🚀 RECOMMENDED NEXT STEPS (Priority Order):")
    print("=" * 60)
    
    steps = [
        {
            "step": "1. Pass consolidated document to agent",
            "description": "Modify agent to receive consolidated document context",
            "files": ["document_agent.py", "ai_reasoning_engine.py"],
            "effort": "Medium",
            "impact": "High"
        },
        {
            "step": "2. Add asset-specific lookup logic",
            "description": "Agent should look up actual asset values instead of using defaults",
            "files": ["document_agent.py"],
            "effort": "Medium",
            "impact": "High"
        },
        {
            "step": "3. Implement validation during correction",
            "description": "Check corrections against ground truth before applying",
            "files": ["document_agent.py", "ai_reasoning_engine.py"],
            "effort": "Low",
            "impact": "Medium"
        },
        {
            "step": "4. Add calculation logic for ratios",
            "description": "Calculate MOIC, IRR from actual asset data instead of defaults",
            "files": ["ai_reasoning_engine.py"],
            "effort": "High",
            "impact": "Medium"
        },
        {
            "step": "5. Enhance agent prompts",
            "description": "Make prompts more specific to asset-level analysis",
            "files": ["ai_reasoning_engine.py"],
            "effort": "Low",
            "impact": "Medium"
        }
    ]
    
    for step in steps:
        print(f"🔧 {step['step']}")
        print(f"   Description: {step['description']}")
        print(f"   Files to modify: {', '.join(step['files'])}")
        print(f"   Effort: {step['effort']}, Impact: {step['impact']}")
        print()

if __name__ == "__main__":
    analyze_correction_patterns()
    identify_improvement_opportunities()
    recommend_next_steps()
    
    print("💡 QUICK WIN:")
    print("   Start with step 1 (pass consolidated document to agent)")
    print("   This will give the agent access to actual data for corrections")
    print("   Expected improvement: 2/121 → 30-50/121 (25-40% improvement)")