import streamlit as st
from datetime import datetime
from typing import List, Dict

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []
    if 'chunks_ready' not in st.session_state:
        st.session_state.chunks_ready = False
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []

def add_to_history(question: str, answer: str, context_chunks: List[Dict]):
    """Add Q&A pair to session history"""
    entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'question': question,
        'answer': answer,
        'context': context_chunks
    }
    st.session_state.qa_history.append(entry)

def format_history_for_download() -> str:
    """Format Q&A history for download"""
    if not st.session_state.qa_history:
        return "No Q&A history available."
    
    formatted_text = "StudyMate Q&A Session History\n"
    formatted_text += "=" * 50 + "\n\n"
    
    for i, entry in enumerate(st.session_state.qa_history, 1):
        formatted_text += f"Q{i}: {entry['question']}\n"
        formatted_text += f"Time: {entry['timestamp']}\n"
        formatted_text += f"Answer: {entry['answer']}\n"
        formatted_text += "-" * 30 + "\n\n"
    
    return formatted_text

def display_qa_history():
    """Display Q&A history in the interface"""
    if st.session_state.qa_history:
        st.subheader("📝 Q&A History")
        for i, entry in enumerate(reversed(st.session_state.qa_history), 1):
            with st.expander(f"Q{len(st.session_state.qa_history) - i + 1}: {entry['question'][:50]}..."):
                st.write(f"**Question:** {entry['question']}")
                st.write(f"**Answer:** {entry['answer']}")
                st.write(f"**Time:** {entry['timestamp']}")