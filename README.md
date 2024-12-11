# MultiPDF - ChatBot: Empowering PDF Conversations

## Overview

**MultiPDF** is a Streamlit-based chatbot application designed to interact with multiple PDF documents. This powerful tool allows users to upload PDF files, extract their content, and ask questions about the information contained within those documents. By utilizing pre-trained language models, MultiPDF generates relevant and accurate responses based on the context extracted from the PDFs.

## Features

- **Multiple PDF Uploads**: Users can upload multiple PDF files for processing in one session.
- **Text Extraction**: Automatically extracts the text content from PDF documents.
- **Text Chunking**: Divides large blocks of text into manageable chunks for efficient processing.
- **FAISS Vector Store**: Utilizes Facebook AI Similarity Search (FAISS) for fast and accurate similarity searches within the document chunks.
- **Model Selection**: Choose between various pre-trained language models (e.g., GPT-based models) for generating responses.
- **Interactive Chat Interface**: A user-friendly chat interface to ask questions and receive answers based on the PDF content.

## Requirements

To run this application, make sure you have the following Python packages installed:

- Python 3.7 or later
- Streamlit
- Transformers
- PyPDF2
- Langchain
- Langchain Community
- Sentence Transformers
- Torch
- NumPy

You can install all the required dependencies using the following command:

```bash
pip install streamlit transformers PyPDF2 langchain langchain-community sentence-transformers torch numpy

```
## How It Works
This code is a Streamlit application that creates a chatbot based on PDF documents. Users can upload one or more PDF files and ask questions based on the extracted text. The app uses pre-loaded language models to generate appropriate responses. Here’s a summary of how it works:

### 1. Libraries Used:
- **streamlit**: For building the web interface.
- **transformers**: For loading and working with language models.
- **PyPDF2**: To extract text from PDFs.
- **langchain**: For text processing and vector creation.
- **sentence_transformers**: For creating text embeddings.
- **FAISS**: For fast vector search.
- **torch**: For deep learning operations.

### 2. `SentenceTransformerEmbedding` Class:
This class converts text into embedding vectors using the SentenceTransformer model.

### 3. `PDFChatAssistant` Class:
The main class that handles the app’s functionality:
- **load_models**: Loads language models like `facebook/opt-350m` and `distilgpt2`.
- **process_pdf**: Extracts text from PDF files using PyPDF2.
- **get_pdf_text**: Uses multi-threading to process multiple PDFs in parallel.
- **get_text_chunks**: Splits the extracted text into manageable chunks.
- **get_vectorstore**: Converts text chunks into vectors and stores them in a FAISS vector store.
- **generate_response**: Generates a response to the user’s question using the selected model.

### 4. `run` Function:
- Starts the user interface and allows PDF uploads and model selection.
- Processes the uploaded PDFs, extracts text, and splits it into chunks.
- Generates responses based on user questions and displays chat history.

### 5. `main` Function:
- Initializes the app by running the `PDFChatAssistant` class.

### 6. Streamlit Interface:
- Users upload PDFs, ask questions, and view responses via the web interface.
- The interface allows model selection, PDF uploads, and question submission.
- The bot’s responses are displayed along with the chat history.

### Workflow Steps:
1. **Model Loading**: The models are loaded initially.
2. **PDF Processing**: Users upload PDFs and extract text.
3. **Text Splitting**: The extracted text is split into chunks suitable for FAISS search.
4. **Q&A**: Users ask questions, and the model generates answers.
5. **Chat History**: Displays the conversation between the user and the bot.

   
# How to Use

### Start the Application:
Run the Streamlit app with the following command:
 ```bash
streamlit run app.py
```
### Upload PDFs:
Once the application is running, upload one or more PDF files via the file uploader in the sidebar.

### Ask Questions:
After the PDFs are processed, you can type a question in the text input box. The chatbot will use the context from the uploaded PDFs to generate an answer.

### Select a Model:
You can select a language model from the dropdown in the sidebar. The models are pre-loaded, and you can choose the one that best suits your needs.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
