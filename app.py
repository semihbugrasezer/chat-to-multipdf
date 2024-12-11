import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np
import os
from htmltemplates import css, user_template, bot_template

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SentenceTransformerEmbedding:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True).tolist()


class PDFChatAssistant:
    def __init__(self):
        st.set_page_config(
            page_title="MultiPDF - ChatBot: Empowering PDF Conversations",
            page_icon="📄",
            layout="wide",
        )

    def load_models(self):
        """Load open-source models."""
        models_to_try = [
            {
                "name": "facebook/opt-350m",
                "model_class": AutoModelForCausalLM,
                "tokenizer_class": AutoTokenizer,
            },
            {
                "name": "distilgpt2",
                "model_class": AutoModelForCausalLM,
                "tokenizer_class": AutoTokenizer,
            },
        ]

        loaded_models = []
        for model_config in models_to_try:
            try:
                model_name = model_config["name"]
                model = model_config["model_class"].from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    pad_token_id=50256,
                )
                tokenizer = model_config["tokenizer_class"].from_pretrained(model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = tokenizer.pad_token_id

                loaded_models.append(
                    {"name": model_name, "model": model, "tokenizer": tokenizer}
                )
                st.success(f"Successfully loaded {model_name}")
            except Exception as e:
                st.warning(f"Failed to load {model_name}: {e}")

        return loaded_models

    def process_pdf(self, pdf):
        """Extract text from PDF."""
        try:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception as e:
            st.warning(f"Error processing PDF: {e}")
            return ""

    def get_pdf_text(self, pdf_docs):
        """Extract text from multiple PDFs in parallel."""
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.process_pdf, pdf_docs))
        return "".join(results)

    def get_text_chunks(self, text, chunk_size=500, chunk_overlap=100):
        """Split text into manageable chunks."""
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        return text_splitter.split_text(text)

    def get_vectorstore(self, text_chunks):
        """Create a FAISS vector store from text chunks using a SentenceTransformer wrapper."""
        embedding_model = SentenceTransformerEmbedding(
            "all-MiniLM-L6-v2"
        )  # Create an embedding model instance

        # Create the FAISS vector store
        vectorstore = FAISS.from_texts(
            text_chunks, embedding=embedding_model  # Pass the wrapper instance
        )
        return vectorstore

    def generate_response(self, question, context, model, tokenizer):
        """Generate response using the selected model."""
        try:
            input_text = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            ).to(model.device)

            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=100,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
            )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.split("Answer:")[-1].strip()
        except Exception as e:
            st.error(f"Response generation error: {e}")
            return "Sorry, I couldn't generate a response."

    def run(self):
        """Main Streamlit application."""
        st.title("📄 MultiPDF - ChatBot: Empowering PDF Conversations")
        st.markdown(css, unsafe_allow_html=True)

        # Add components to the sidebar
        st.sidebar.title("Options")
        st.sidebar.markdown("Select models and upload PDFs")

        models = self.load_models()
        if not models:
            st.error("No models could be loaded. Please check your setup.")
            return

        pdf_docs = st.sidebar.file_uploader(
            "Upload PDF files", type=["pdf"], accept_multiple_files=True
        )

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        if pdf_docs:
            with st.spinner("Processing PDFs..."):
                raw_text = self.get_pdf_text(pdf_docs)
                text_chunks = self.get_text_chunks(raw_text)
                vectorstore = self.get_vectorstore(text_chunks)

                if vectorstore:
                    st.success(
                        f"Processed {len(text_chunks)} text chunks from {len(pdf_docs)} PDFs."
                    )

        user_question = st.text_input("Ask a question about your documents:")

        selected_model = st.sidebar.selectbox(
            "Choose Model", [model["name"] for model in models]
        )
        current_model = next(
            (m for m in models if m["name"] == selected_model), models[0]
        )

        if user_question:
            response = self.generate_response(
                user_question,
                raw_text,
                current_model["model"],
                current_model["tokenizer"],
            )
            st.session_state.chat_history.append(("user", user_question))
            st.session_state.chat_history.append(("bot", response))

        for sender, message in st.session_state.chat_history:
            if sender == "user":
                st.markdown(user_template.format(message), unsafe_allow_html=True)
            else:
                st.markdown(bot_template.format(message), unsafe_allow_html=True)


def main():
    assistant = PDFChatAssistant()
    assistant.run()


if __name__ == "__main__":
    main()
