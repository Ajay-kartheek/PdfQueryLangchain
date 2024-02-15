import gradio as gr
import os
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# Replace with your API key
OPENAI_API_KEY = "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"

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def pdf_reader_chat_response(pdf_path, query):
    """Processes a PDF, uses Langchain for knowledge retrieval, and provides an OpenAI-powered chat response."""

    try:
        # Read PDF content
        with open(pdf_path, "rb") as f:
            pdfreader = PdfReader(f)
            raw_text = ""
            for page in pdfreader.pages:
                content = page.extract_text()
                if content:
                    raw_text += content

        # Text processing and knowledge retrieval
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=200)
        texts = text_splitter.split_text(raw_text)
        embeddings = OpenAIEmbeddings()
        document_search = FAISS.from_texts(texts, embeddings)

        # Question answering with chain
        chain = load_qa_chain(OpenAI(), chain_type="stuff")
        docs = document_search.similarity_search(query)
        answer = chain.run(input_documents=docs, question=query)

        return answer

    except Exception as e:
        return f"Error: {e}"


interface = gr.Interface(
    fn=pdf_reader_chat_response,
    inputs=[
        gr.File(label="Upload PDF"),
        gr.Textbox(label="Ask a question"),
    ],
    outputs=[gr.Textbox(label="Answer")],
    title="PDF Whisperer... ",
    description="Get answers to your questions from the uploaded PDF.",
)

# Launch Gradio app
interface.launch(share=True)