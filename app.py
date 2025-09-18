import streamlit as st
from pdf_processor import PDFProcessor
from retrieval_engine import RetrievalEngine
from llm_handler import FastHuggingFaceHandler  # Use the fast handler
from utils import initialize_session_state, add_to_history, format_history_for_download, display_qa_history

st.set_page_config(
    page_title="StudyMate - Fast AI Assistant", 
    page_icon="⚡",
    layout="wide"
)

# Custom CSS for speed theme
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #FF6B35;
    font-size: 3em;
    margin-bottom: 1em;
}
.speed-box {
    background-color: #FFF3E0;
    padding: 1em;
    border-radius: 10px;
    border-left: 4px solid #FF6B35;
    margin: 1em 0;
}
.fast-answer {
    background-color: #E8F5E8;
    padding: 1em;
    border-radius: 8px;
    border-left: 3px solid #4CAF50;
}
</style>
""", unsafe_allow_html=True)

def main():
    initialize_session_state()
    
    # Initialize components
    if 'pdf_processor' not in st.session_state:
        st.session_state.pdf_processor = PDFProcessor(chunk_size=200, overlap=30)  # Smaller chunks for speed
    
    if 'retrieval_engine' not in st.session_state:
        st.session_state.retrieval_engine = RetrievalEngine()
    
    if 'llm_handler' not in st.session_state:
        with st.spinner("⚡ Loading fast AI model..."):
            st.session_state.llm_handler = FastHuggingFaceHandler()
    
    # Header
    st.markdown('<h1 class="main-header">⚡ StudyMate - Fast Edition</h1>', unsafe_allow_html=True)
    st.markdown("### Lightning-Fast AI Assistant")
    
    # Speed info
    st.markdown("""
    <div class="speed-box">
    <b>⚡ Optimized for Speed</b><br>
    • Lightweight models (~100-500MB)<br>
    • Fast response times (1-3 seconds)<br>
    • Low memory usage<br>
    • Works on any computer
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf", 
            accept_multiple_files=True,
            help="Upload study materials"
        )
        
        if uploaded_files:
            if st.button("⚡ Process PDFs", type="primary"):
                process_pdfs(uploaded_files)
        
        # Model info
        if st.session_state.llm_handler:
            st.markdown("---")
            st.markdown("**🤖 AI Model**")
            model_type = getattr(st.session_state.llm_handler, 'model_type', 'Unknown')
            st.write(f"Type: {model_type.title()}")
            st.write("Status: Ready ⚡")
    
    # Main interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("❓ Ask Your Question")
        question = st.text_input(
            "Question:",
            placeholder="What are the main points about...?",
            key="question_input"
        )
        
        col_btn, col_info = st.columns([1, 2])
        with col_btn:
            ask_btn = st.button("⚡ Get Answer", type="primary", disabled=not st.session_state.chunks_ready)
        with col_info:
            if st.session_state.chunks_ready:
                st.success("Ready to answer!")
            else:
                st.info("Upload PDFs first")
        
        if ask_btn and question.strip():
            get_fast_answer(question)
        
        # Display answer
        if hasattr(st.session_state, 'current_answer'):
            st.markdown("### 💡 Answer")
            st.markdown(f'<div class="fast-answer">{st.session_state.current_answer}</div>', 
                       unsafe_allow_html=True)
            
            # Show sources
            if hasattr(st.session_state, 'current_context'):
                with st.expander("📖 Sources"):
                    for i, chunk in enumerate(st.session_state.current_context, 1):
                        st.write(f"**Source {i}:** {chunk['source']}")
                        st.caption(f"{chunk['text'][:150]}...")
    
    with col2:
        st.subheader("📊 Status")
        if st.session_state.chunks_ready:
            st.metric("Files Processed", len(st.session_state.processed_files))
            st.metric("Ready", "Yes ✅")
        else:
            st.metric("Files Processed", 0)
            st.metric("Ready", "No 📤")
        
        # Quick stats
        if st.session_state.qa_history:
            st.metric("Questions Asked", len(st.session_state.qa_history))
            
            # Download button
            st.markdown("---")
            history_text = format_history_for_download()
            st.download_button(
                "📥 Download History",
                history_text,
                "studymate_fast_history.txt"
            )
    
    # Q&A History
    display_qa_history()

def process_pdfs(uploaded_files):
    """Fast PDF processing"""
    with st.spinner("⚡ Processing PDFs..."):
        try:
            chunks = st.session_state.pdf_processor.process_multiple_pdfs(uploaded_files)
            
            # 🔍 ADD THIS DEBUG CODE HERE:
            print(f"📄 DEBUG: PDF processing created {len(chunks) if chunks else 0} chunks")
            if chunks:
                print(f"📄 DEBUG: First chunk sample: {chunks[0]['text'][:100]}...")
                print(f"📄 DEBUG: First chunk source: {chunks[0].get('source', 'Unknown')}")
            else:
                print("❌ DEBUG: PDF processing returned no chunks!")
            
            if chunks:
                st.session_state.retrieval_engine.build_index(chunks)
                
                # 🔍 ADD THIS TOO:
                print(f"🔍 DEBUG: FAISS index built with {len(chunks)} chunks")
                
                st.session_state.chunks_ready = True
                st.session_state.processed_files = [f.name for f in uploaded_files]
                st.success(f"⚡ Processed {len(uploaded_files)} files in seconds! ({len(chunks)} chunks)")
            else:
                st.error("No text found in PDFs")
                
        except Exception as e:
            print(f"❌ DEBUG: Error in process_pdfs: {str(e)}")
            st.error(f"Processing error: {str(e)}")


def get_fast_answer(question):
    """Get fast answer"""
    with st.spinner("⚡ Generating answer..."):
        try:
            # Fast retrieval
            chunks = st.session_state.retrieval_engine.retrieve_relevant_chunks(question, k=3)
            
            # 🔍 ADD THIS DEBUG CODE HERE:
            print(f"🔍 DEBUG: Retrieved chunks: {len(chunks)}")
            for i, chunk in enumerate(chunks):
                print(f"🔍 DEBUG: Chunk {i}: {chunk['text'][:50]}...")
                print(f"🔍 DEBUG: Chunk source: {chunk.get('source', 'Unknown')}")
            
            if chunks:
                # Fast answer generation
                answer = st.session_state.llm_handler.generate_answer(question, chunks)
                
                # 🔍 ADD THIS DEBUG TOO:
                print(f"🤖 DEBUG: Generated answer: {answer}")
                
                # Store results
                st.session_state.current_answer = answer
                st.session_state.current_context = chunks
                
                # Add to history
                add_to_history(question, answer, chunks)
                st.balloons()  # Celebration for speed!
            else:
                # 🔍 AND THIS:
                print("❌ DEBUG: No chunks retrieved!")
                st.warning("No relevant content found")
                
        except Exception as e:
            print(f"❌ DEBUG: Error in get_fast_answer: {str(e)}")
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()