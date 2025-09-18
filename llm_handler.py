from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from typing import List, Dict
import os
from dotenv import load_dotenv

load_dotenv()

class FastHuggingFaceHandler:
    def __init__(self):
        """Initialize with a fast, lightweight model"""
        self.model_type = "qa"
        self.setup_qa_model()
    
    def setup_qa_model(self):
        """Setup extractive Q&A model (fastest option)"""
        try:
            print("Loading fast Q&A model...")
            self.qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                device=-1
            )
            self.model_type = "qa"
            print("✅ Fast Q&A model loaded successfully!")
        except Exception as e:
            print(f"❌ Q&A model failed: {e}")
            self.setup_simple_fallback()
    
    def setup_simple_fallback(self):
        """Simple text generation fallback"""
        try:
            print("Loading fallback model...")
            self.qa_pipeline = pipeline(
                "text2text-generation",
                model="google/flan-t5-small",
                device=-1,
                max_length=200
            )
            self.model_type = "fallback"
            print("✅ Fallback model loaded!")
        except Exception as e:
            print(f"❌ All models failed: {e}")
            self.qa_pipeline = None
            self.model_type = "none"
    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate answer using the loaded model"""
        print(f"🤖 DEBUG: LLM generate_answer called")
        print(f"🤖 DEBUG: Query: {query}")
        print(f"🤖 DEBUG: Number of context chunks: {len(context_chunks)}")
        
        if not context_chunks:
            print("❌ DEBUG: No context chunks provided to LLM!")
            return "No relevant context found in documents. Please check if PDFs were processed correctly."
        
        # Print context for debugging
        for i, chunk in enumerate(context_chunks):
            print(f"🤖 DEBUG: Chunk {i}: {chunk['text'][:100]}...")
        
        if not hasattr(self, 'qa_pipeline') or not self.qa_pipeline:
            return "❌ Model not available. Please restart the application."
        
        try:
            if self.model_type == "qa":
                return self.qa_answer(query, context_chunks)
            else:
                return self.fallback_answer(query, context_chunks)
        except Exception as e:
            error_msg = f"Error generating answer: {str(e)}"
            print(f"❌ DEBUG: {error_msg}")
            return error_msg
    
    def qa_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """Use extractive Q&A model with improved context handling"""
        try:
            # Combine context more effectively
            contexts = []
            for chunk in context_chunks:
                chunk_text = chunk['text'].strip()
                if chunk_text and len(chunk_text) > 10:  # Only meaningful chunks
                    contexts.append(chunk_text[:400])  # Increased chunk size
            
            if not contexts:
                return "No valid text found in the documents."
            
            # Better context combination
            context = " ... ".join(contexts)
            
            # Increased context limit
            if len(context) > 3000:  # Much higher limit
                context = context[:3000] + "..."
            
            print(f"🤖 DEBUG: Final context ({len(context)} chars): {context[:200]}...")
            
            result = self.qa_pipeline(question=query, context=context)
            answer = result['answer'].strip()
            confidence = result['score']
            
            print(f"🤖 DEBUG: Raw answer: '{answer}'")
            print(f"🤖 DEBUG: Confidence: {confidence}")
            
            # Much more lenient confidence threshold
            if answer and len(answer) > 1 and confidence > 0.001:  # Very low threshold
                # Clean up the answer
                if len(answer) > 200:
                    answer = answer[:200] + "..."
                return answer
            else:
                # Try to extract any relevant information
                return f"Based on available text: {answer} (Low confidence - please verify)"
                
        except Exception as e:
            print(f"❌ QA Error: {e}")
            return f"Error processing question: {str(e)}"
    
    def fallback_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """Fallback method using generative model"""
        try:
            context = " ".join([chunk['text'][:200] for chunk in context_chunks if chunk['text'].strip()])
            prompt = f"Based on this text, answer the question: {query}\n\nText: {context}\n\nAnswer:"
            
            result = self.qa_pipeline(prompt, max_length=150, temperature=0.7)
            
            if result and len(result) > 0:
                answer = result[0]['generated_text'].strip()
                # Remove the prompt from answer if it's included
                if prompt in answer:
                    answer = answer.replace(prompt, "").strip()
                return answer if len(answer) > 10 else "Could not generate a clear answer."
            else:
                return "Could not generate answer from the provided text."
                
        except Exception as e:
            return f"Fallback error: {str(e)}"