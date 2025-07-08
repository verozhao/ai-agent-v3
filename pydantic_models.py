"""
Generic Pydantic Models for Any Financial Document Structure
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, date
from decimal import Decimal

class GenericAssetModel(BaseModel):
    """Generic asset model that can handle any asset structure"""
    name: Optional[str] = Field(None, description="Asset name or identifier")
    
    class Config:
        extra = "allow"  # Allow any additional fields
        
    @field_validator('*', mode='before')
    @classmethod
    def validate_numeric_fields(cls, v, info):
        """Validate common numeric fields to prevent negative values where inappropriate"""
        if info.field_name in ['total_invested', 'current_value', 'realized_value', 'unrealized_value', 'total_value']:
            if v is not None and isinstance(v, (int, float)) and v < 0:
                return 0.0  # Don't allow negative values for these fields
        return v

class ParsedDocumentModel(BaseModel):
    """Generic model for any parsed document structure"""
    
    # Common fields that most financial documents have
    document_path: Optional[str] = Field(None, description="Source document path")
    document_type: Optional[str] = Field(None, description="Type of document")
    reporting_date: Optional[Union[str, date]] = Field(None, description="As-of date for the document")
    currency: Optional[str] = Field(None, description="Currency used in the document")
    
    # Optional organization identification
    fund_name: Optional[str] = Field(None, description="Fund or entity name")
    fund_org_id: Optional[str] = Field(None, description="Organization identifier")
    
    # Generic assets list that can handle any structure
    assets: Optional[List[Union[GenericAssetModel, Dict[str, Any]]]] = Field(default_factory=list, description="Assets or investments")
    
    class Config:
        extra = "allow"  # Allow any additional fields from the parsed document
        
    @model_validator(mode='before')
    @classmethod
    def parse_dynamic_structure(cls, values):
        """Parse and validate the dynamic document structure"""
        # Handle assets in various formats
        if 'assets' in values and values['assets']:
            assets = values['assets']
            
            if isinstance(assets, dict):
                # Convert dict format to list
                asset_list = []
                for asset_name, asset_data in assets.items():
                    if isinstance(asset_data, dict):
                        asset_data_copy = asset_data.copy()
                        asset_data_copy['name'] = asset_name
                        asset_list.append(asset_data_copy)
                    else:
                        asset_list.append({'name': asset_name, 'data': asset_data})
                values['assets'] = asset_list
            
            elif isinstance(assets, list):
                # Ensure each asset has at least a name
                for i, asset in enumerate(assets):
                    if isinstance(asset, dict) and 'name' not in asset:
                        # Try to extract name from other fields
                        name = asset.get('company_name') or asset.get('asset_name') or f"Asset_{i+1}"
                        asset['name'] = name
        
        return values
    
    @field_validator('fund_name')
    @classmethod
    def validate_fund_name(cls, v):
        """Basic validation for fund name"""
        if v and v.strip():
            import re
            # Don't allow obvious date patterns as fund names
            if re.match(r'^\d{4}-\d{2}-\d{2}$', v.strip()):
                raise ValueError(f"Fund name appears to be a date: {v}")
        return v
    
    def get_financial_fields(self) -> List[str]:
        """Get all fields that appear to be financial/numeric"""
        financial_fields = []
        
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, (int, float)):
                financial_fields.append(field_name)
            elif field_name.lower().endswith(('_value', '_amount', '_capital', '_total', '_invested', '_irr', '_multiple')):
                financial_fields.append(field_name)
                
        return financial_fields
    
    def get_asset_count(self) -> int:
        """Get number of assets in the document"""
        if not self.assets:
            return 0
        return len(self.assets)
    
    def validate_financial_consistency(self) -> List[str]:
        """Generic financial validation"""
        issues = []
        
        # Check for negative values in obvious financial fields
        financial_fields = self.get_financial_fields()
        
        for field_name in financial_fields:
            value = getattr(self, field_name, None)
            if value is not None and isinstance(value, (int, float)) and value < 0:
                # Some fields can legitimately be negative (like returns)
                if not any(keyword in field_name.lower() for keyword in ['irr', 'return', 'gain', 'loss', 'delta']):
                    issues.append(f"{field_name} has negative value: {value}")
        
        # Asset-level validation
        if self.assets:
            for i, asset in enumerate(self.assets):
                if isinstance(asset, dict):
                    for field, value in asset.items():
                        if isinstance(value, (int, float)) and value < 0:
                            if not any(keyword in field.lower() for keyword in ['irr', 'return', 'gain', 'loss', 'delta']):
                                issues.append(f"Asset {i+1} ({asset.get('name', 'unnamed')}) has negative {field}: {value}")
        
        return issues

class ConsolidatedDocumentModel(BaseModel):
    """Generic model for consolidated document validation"""
    fund_org_id: str
    reporting_date: Union[str, date]
    document_type: str = "consolidated"
    
    # Dynamic data structure
    data: Dict[str, Any] = Field(default_factory=dict, description="Consolidated document data")
    
    # Validation metadata
    validation_date: datetime = Field(default_factory=datetime.now)
    validation_source: str = "consolidated_database"
    
    class Config:
        extra = "allow"

# Factory functions
def create_document_model_from_parsed_document(parsed_doc: Dict[str, Any]) -> ParsedDocumentModel:
    """Create generic Pydantic model from any parsed document structure"""
    return ParsedDocumentModel(**parsed_doc)

def validate_corrected_document(corrected_doc: Dict[str, Any]) -> ParsedDocumentModel:
    """Validate corrected document using generic Pydantic model"""
    model = ParsedDocumentModel(**corrected_doc)
    
    # Run additional validation
    issues = model.validate_financial_consistency()
    if issues:
        print(f"Validation issues found: {issues}")
    
    return model

# Backward compatibility aliases
def create_pe_fund_model_from_parsed_document(parsed_doc: Dict[str, Any]) -> ParsedDocumentModel:
    """Backward compatibility - now uses generic model"""
    return create_document_model_from_parsed_document(parsed_doc)

# Export
__all__ = [
    "ParsedDocumentModel", 
    "GenericAssetModel", 
    "ConsolidatedDocumentModel",
    "create_document_model_from_parsed_document",
    "create_pe_fund_model_from_parsed_document",  # Backward compatibility
    "validate_corrected_document"
]