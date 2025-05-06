import os
import requests
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from typing import Optional, List
import tempfile

# === Page Configuration ===
st.set_page_config(
    page_title="PDF Q&A Assistant",
    page_icon="📚",
    layout="wide"
)

# === Set your Groq API Key ===
GROQ_API_KEY = "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"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# === Custom LLM wrapper for Groq API ===
class GroqLLM(LLM):
    model: str = "llama3-8b-8192"
    temperature: float = 0.2
    groq_api_key: str = os.environ.get("GROQ_API_KEY")
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
            }
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()  # Raise exception for HTTP errors
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            st.error(f"Error with Groq API: {str(e)}")
            return f"Error: {str(e)}"
    
    @property
    def _llm_type(self) -> str:  # Fixed syntax
        return "groq"

# === Custom Prompt Template ===
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Use the following context to answer the user's question.
Context:
{context}
Question:
{question}
Answer the question in a detailed and accurate way.
"""
)

# === Function to Load and Process PDF ===
def process_pdf(pdf_path):
    try:
        with st.spinner("Loading PDF..."):
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            st.success(f"✅ Loaded {len(documents)} pages from PDF")
            
        with st.spinner("Splitting text into chunks..."):
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_docs = text_splitter.split_documents(documents)
            st.success(f"✅ Split into {len(split_docs)} text chunks")
            
        with st.spinner("Creating vector embeddings... (this may take a while)"):
            embedding = HuggingFaceEmbeddings(model_name="bert-base-uncased")
            vectordb = Chroma.from_documents(split_docs, embedding=embedding)
            retriever = vectordb.as_retriever()
            st.success("✅ Vector database created successfully")
            
        with st.spinner("Setting up Q&A chain..."):
            llm = GroqLLM()
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": custom_prompt},
                return_source_documents=True
            )
            st.success("✅ Q&A system ready!")
            
        return qa_chain
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

# === Streamlit App Layout ===
def main():
    st.title("📚 PDF Question Answering System")
    st.write("Upload a PDF and ask questions about its content. Powered by LangChain and Groq LLM.")
    
    # Initialize session state
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'pdf_name' not in st.session_state:
        st.session_state.pdf_name = ""
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Option to use default PDF path or upload a new one
        pdf_option = st.radio("PDF Source", ["Upload New PDF", "Use Default Path"])
        
        if pdf_option == "Upload New PDF":
            uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
            if uploaded_file and (not st.session_state.pdf_processed or uploaded_file.name != st.session_state.pdf_name):
                # Save uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                    temp_pdf.write(uploaded_file.getbuffer())
                    pdf_path = temp_pdf.name
                
                st.session_state.qa_chain = process_pdf(pdf_path)
                st.session_state.pdf_processed = st.session_state.qa_chain is not None
                st.session_state.pdf_name = uploaded_file.name
                
                # Clean up temporary file
                try:
                    os.unlink(pdf_path)
                except:
                    pass
        else:
            default_path = st.text_input("Enter PDF Path", "C:/Users/anjuk/OneDrive/Desktop/Datasense/MLBOOK (1).pdf")
            process_button = st.button("Process PDF")
            
            if process_button and default_path and (not st.session_state.pdf_processed or default_path != st.session_state.pdf_name):
                if os.path.exists(default_path):
                    st.session_state.qa_chain = process_pdf(default_path)
                    st.session_state.pdf_processed = st.session_state.qa_chain is not None
                    st.session_state.pdf_name = default_path
                else:
                    st.error(f"File not found: {default_path}")
        
        # Reset button
        if st.button("Reset"):
            st.session_state.qa_chain = None
            st.session_state.pdf_processed = False
            st.session_state.pdf_name = ""
            st.success("Reset successful! Please upload a new PDF.")
    
    # Main area
    if st.session_state.pdf_processed and st.session_state.qa_chain:
        st.success(f"Currently using PDF: {st.session_state.pdf_name}")
        
        # Query input
        query = st.text_input("Ask a question about your PDF:")
        
        if query:
            with st.spinner("Generating answer..."):
                try:
                    # Invoke the QA chain
                    response = st.session_state.qa_chain.invoke({"query": query})
                    
                    # Extract answer and source documents
                    answer = response.get("result", "No answer found.")
                    source_docs = response.get("source_documents", [])
                    
                    # Display answer
                    st.header("Answer")
                    st.write(answer)
                    
                    # Display source documents
                    if source_docs:
                        st.header("Source Documents")
                        for i, doc in enumerate(source_docs):
                            with st.expander(f"Source {i+1}"):
                                st.write(doc.page_content)
                                if hasattr(doc, 'metadata') and doc.metadata:
                                    st.write(f"Page: {doc.metadata.get('page', 'Unknown')}")
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
                    
                    # Fallback to direct LLM call
                    st.write("Attempting direct LLM response...")
                    try:
                        llm = GroqLLM()
                        direct_answer = llm._call(f"Answer this question based on your knowledge: {query}")
                        st.write("Direct LLM Response:")
                        st.write(direct_answer)
                    except:
                        st.error("Failed to get fallback response.")
    else:
        if not st.session_state.pdf_processed:
            st.info("Please upload a PDF or provide a valid path to start.")
        elif st.session_state.qa_chain is None:
            st.warning("PDF processing failed. Please try again with a different PDF.")

if __name__ == "__main__":
    main()