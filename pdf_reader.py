import pdfplumber
import fitz
from typing import Dict, Any, List, Optional

class PDFDocumentReader:
    """Direct PDF reading for validation"""
    
    async def read_pdf_for_validation(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text and tables from PDF for cross-validation"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Extract tables that might contain financial data
                all_tables = []
                for page in pdf.pages:
                    tables = page.extract_tables()
                    if tables:
                        all_tables.extend(tables)
                
                # Extract text for context
                full_text = "\n".join(page.extract_text() for page in pdf.pages)
                
                return {
                    "tables": all_tables,
                    "text": full_text,
                    "page_count": len(pdf.pages)
                }
        except Exception as e:
            logger.error(f"PDF reading failed: {e}")
            return None