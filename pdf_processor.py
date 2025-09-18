import PyPDF2
import re
from typing import List, Dict
import io

class PDFProcessor:
    def __init__(self, chunk_size: int = 400, overlap: int = 100):  # Larger chunks
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file using PyPDF2"""
        try:
            pdf_file.seek(0)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            print(f"📄 DEBUG: PDF has {len(pdf_reader.pages)} pages")
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                print(f"📄 DEBUG: Page {page_num+1} extracted {len(page_text)} characters")
                if page_text.strip():  # Only add if page has content
                    text += page_text + " "
            
            print(f"📄 DEBUG: Total extracted text: {len(text)} characters")
            print(f"📄 DEBUG: Text sample: {text[:200]}...")
            
            cleaned_text = self.clean_text(text)
            print(f"📄 DEBUG: After cleaning: {len(cleaned_text)} characters")
            
            return cleaned_text
        except Exception as e:
            print(f"❌ PDF extraction error: {str(e)}")
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text - less aggressive cleaning"""
        # Remove excessive whitespace but preserve structure
        text = re.sub(r'\s+', ' ', text)
        # Only remove truly problematic characters, keep most punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\'\"\%\$\@\#]', ' ', text)
        return text.strip()
    
    def create_chunks(self, text: str, filename: str) -> List[Dict]:
        """Split text into overlapping chunks"""
        if not text or len(text.strip()) < 20:
            print(f"❌ DEBUG: Text too short for chunking: {len(text)} chars")
            return []
            
        words = text.split()
        chunks = []
        
        print(f"📝 DEBUG: Creating chunks from {len(words)} words")
        
        for i in range(0, len(words), max(1, self.chunk_size - self.overlap)):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            if len(chunk_text.strip()) > 50:  # Only keep substantial chunks
                chunks.append({
                    'text': chunk_text,
                    'source': filename,
                    'chunk_id': len(chunks)
                })
                print(f"📝 DEBUG: Created chunk {len(chunks)}: {chunk_text[:100]}...")
            
            if i + self.chunk_size >= len(words):
                break
        
        print(f"📝 DEBUG: Created {len(chunks)} total chunks")
        return chunks
    
    def process_multiple_pdfs(self, pdf_files) -> List[Dict]:
        """Process multiple PDF files and return combined chunks"""
        all_chunks = []
        
        for pdf_file in pdf_files:
            try:
                print(f"🔄 Processing {pdf_file.name}...")
                text = self.extract_text_from_pdf(pdf_file)
                
                if text and len(text.strip()) > 100:  # Ensure meaningful content
                    chunks = self.create_chunks(text, pdf_file.name)
                    all_chunks.extend(chunks)
                    print(f"✅ Processed {pdf_file.name}: {len(chunks)} chunks")
                else:
                    print(f"⚠️ {pdf_file.name}: Insufficient text extracted")
                    
            except Exception as e:
                print(f"❌ Error processing {pdf_file.name}: {str(e)}")
                continue
        
        print(f"🎯 Total chunks created: {len(all_chunks)}")
        return all_chunks