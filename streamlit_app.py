import streamlit as st
import os
import tempfile
import uuid
from typing import List, Dict, Any
from pathlib import Path

# Core libraries
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Fixed imports - using the correct module paths for newer LangChain versions
try:
    # Try newer import path first
    from langchain.chains.history_aware_retriever import create_history_aware_retriever
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    # Fallback to alternative import locations
    try:
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser
        
        # We'll need to create custom implementations if imports fail
        def create_history_aware_retriever(llm, retriever, prompt):
            """Fallback implementation"""
            from langchain_core.runnables import RunnableBranch, RunnableLambda
            
            def has_chat_history(input_dict):
                return len(input_dict.get("chat_history", [])) > 0
            
            contextualized_question = (
                prompt 
                | llm 
                | StrOutputParser()
            )
            
            return RunnableBranch(
                (
                    RunnableLambda(has_chat_history),
                    contextualized_question | retriever
                ),
                RunnableLambda(lambda x: x["input"]) | retriever
            )
        
        def create_retrieval_chain(retriever, combine_docs_chain):
            """Fallback implementation"""
            from langchain_core.runnables import RunnableParallel, RunnablePassthrough
            
            return RunnableParallel(
                {
                    "context": retriever,
                    "input": RunnablePassthrough()
                }
            ) | combine_docs_chain
        
        def create_stuff_documents_chain(llm, prompt):
            """Fallback implementation"""
            from langchain_core.runnables import RunnableParallel
            
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            return (
                RunnableParallel(
                    {
                        "context": lambda x: format_docs(x["context"]),
                        "input": lambda x: x["input"],
                        "chat_history": lambda x: x.get("chat_history", [])
                    }
                )
                | prompt
                | llm
                | StrOutputParser()
                | (lambda x: {"answer": x})
            )
    except Exception as e:
        st.error(f"Failed to import required LangChain components: {e}")
        st.stop()


# Page configuration
st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None

# Chat message history management
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.chat_history:
        st.session_state.chat_history[session_id] = ChatMessageHistory()
    return st.session_state.chat_history[session_id]

@st.cache_resource
def create_embeddings():
    """Create HuggingFace embeddings with caching"""
    try:
        # Try to get HF token from secrets, otherwise use default
        hf_token = st.secrets.get("hf_token", None)
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return None

@st.cache_resource
def create_vector_store(pdf_files, _embeddings):
    """Create and cache vector store from PDF files"""
    if not pdf_files or not _embeddings:
        return None
    
    try:
        # Process PDFs
        all_documents = []
        
        for pdf_file in pdf_files:
            # Create temporary file with unique name
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, 
                suffix=".pdf",
                prefix=f"uploaded_pdf_{uuid.uuid4().hex}_"
            )
            temp_file.write(pdf_file.read())
            temp_file.close()
            
            try:
                # Load PDF
                loader = PyPDFLoader(temp_file.name)
                documents = loader.load()
                all_documents.extend(documents)
            except Exception as e:
                st.error(f"Error loading PDF {pdf_file.name}: {str(e)}")
            finally:
                # Clean up temporary file
                os.unlink(temp_file.name)
        
        if not all_documents:
            return None
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=200,
            length_function=len
        )
        splits = text_splitter.split_documents(all_documents)
        
        # Create vector store using FAISS
        vector_store = FAISS.from_documents(
            documents=splits,
            embedding=_embeddings
        )
        
        return vector_store
        
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def create_conversation_chain(vector_store, groq_api_key):
    """Create conversational RAG chain"""
    try:
        # Initialize Groq LLM
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="gemma2-9b-it",
            temperature=0.1
        )
        
        # Create retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6}
        )
        
        # Contextualize question prompt
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        
        # QA system prompt
        qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.

{context}"""
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        # Create conversational chain with history
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        
        return conversational_rag_chain
        
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None

# Main UI
st.title("🤖 RAG Chat Assistant")
st.write("Upload PDFs and chat with your documents using AI")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # API Key input
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Enter your Groq API key to use the Gemma2-9b-It model"
    )
    
    # Session ID input
    session_id = st.text_input(
        "Session ID",
        value="default",
        help="Enter a session ID to maintain separate conversation histories"
    )
    
    # PDF Upload
    st.header("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF files to create a knowledge base"
    )
    
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} file(s)")
        
        # Process PDFs button
        if st.button("Process PDFs", type="primary"):
            if not groq_api_key:
                st.warning("Please enter your Groq API key first")
            else:
                with st.status("Processing PDFs...") as status:
                    # Create embeddings
                    status.update(label="Creating embeddings...")
                    embeddings = create_embeddings()
                    
                    if embeddings:
                        # Create vector store
                        status.update(label="Creating vector store...")
                        vector_store = create_vector_store(uploaded_files, embeddings)
                        
                        if vector_store:
                            # Create conversation chain
                            status.update(label="Setting up conversation chain...")
                            conversation_chain = create_conversation_chain(vector_store, groq_api_key)
                            
                            if conversation_chain:
                                st.session_state.vector_store = vector_store
                                st.session_state.conversation_chain = conversation_chain
                                status.update(label="Ready for chat!", state="complete")
                                st.success("PDFs processed successfully! You can now start chatting.")
                            else:
                                status.update(label="Failed to create conversation chain", state="error")
                        else:
                            status.update(label="Failed to create vector store", state="error")
                    else:
                        status.update(label="Failed to create embeddings", state="error")
    
    # Technical details expander
    with st.expander("Technical Details"):
        st.write("""
        **Models Used:**
        - Embeddings: sentence-transformers/all-MiniLM-L6-v2
        - LLM: Gemma2-9b-It (via Groq)
        - Vector Store: FAISS (in-memory)
        
        **Text Processing:**
        - Chunk Size: 5000 characters
        - Chunk Overlap: 200 characters
        - Retrieval: Top 6 similar chunks
        
        **Features:**
        - History-aware contextualization
        - Persistent session management
        - Multi-file PDF support
        """)

# Main chat interface
if not groq_api_key:
    st.warning("⚠️ Please enter your Groq API key in the sidebar to get started")
    st.info("You can get a free API key from [Groq Console](https://console.groq.com/)")
elif not st.session_state.conversation_chain:
    st.info("📄 Please upload and process PDF files to start chatting")
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get response from conversation chain
                    response = st.session_state.conversation_chain.invoke(
                        {"input": prompt},
                        config={"configurable": {"session_id": session_id}}
                    )
                    
                    # Extract answer from response
                    answer = response.get("answer", "I couldn't generate a response.")
                    
                    # Display response
                    st.markdown(answer)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        if session_id in st.session_state.chat_history:
            st.session_state.chat_history[session_id] = ChatMessageHistory()
        st.rerun()

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, LangChain, and Groq")
