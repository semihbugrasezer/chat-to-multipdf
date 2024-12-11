

## Multipdf to Chat

### 1. **Streamlit**:
   - Used to create the web interface. Users upload PDF files, ask questions, and view the answers.
   - Provides the chatbot interface: Users can type questions and receive answers from the chatbot.

### 2. **PyPDF2**:
   - Used to extract text from the uploaded PDF files. It reads text from the pages of the PDFs.

### 3. **Langchain**:
   - **Text Splitter**: The **RecursiveCharacterTextSplitter** is used to split large text files into meaningful chunks. The text is divided into smaller, more manageable pieces.
   - **GoogleGenerativeAIEmbeddings**: Creates vectors from the text chunks. These vectors are later used for similarity searches.
   - **FAISS**: A library used for efficient and fast vector searches. FAISS stores the vectors of the uploaded text and performs similarity searches using these vectors.
   - **ChatGoogleGenerativeAI**: Generates answers to questions using Google's **Gemini Pro** model.
   - **load_qa_chain**: Loads the question-answer chain, which handles processing the text and generating meaningful answers for the user's questions.

### 4. **dotenv**:
   - Ensures that sensitive information, such as the **Google API Key**, is securely loaded.

## Steps of the Application:

### 1. **PDF Upload**:
   - The user uploads the PDF files. These files are uploaded using `st.file_uploader`.
   - The uploaded PDFs are processed using the `get_pdf_text` function to extract the full text.

### 2. **Text Splitting**:
   - The extracted text is split into chunks of 10,000 characters using the `get_text_chunks` function. This splitting process helps in efficiently processing large texts.

### 3. **Vector Storage**:
   - Vectors are created from the text chunks using **GoogleGenerativeAIEmbeddings**.
   - These vectors are stored in **FAISS**, which enables fast access for similarity searches.

### 4. **Question-Answer Chain**:
   - When the user asks a question, the application searches for answers based on the similarity between the question and the text in the PDFs.
   - Similar texts are retrieved using **FAISS** and answers are generated with **ChatGoogleGenerativeAI**.

### 5. **Streamlit Chat Interface**:
   - The application provides an interactive chat interface between the user and the chatbot. As the user types questions, the chatbot provides appropriate responses.

### 6. **API Key**:
   - The Google API key is read from the `.env` file, allowing the use of Google's Gemini model.

## User Flow:

1. The user uploads the PDF files.
2. The application extracts text from the PDFs, splits it into chunks, and creates vectors.
3. When the user asks a question, the chatbot performs a similarity search on the text.
4. The chatbot uses the Google Gemini model to generate a response and displays the answer to the user.

## Technologies Used:
- **Streamlit**: For the web interface.
- **PyPDF2**: For extracting text from PDFs.
- **Langchain**: For text processing, embedding creation, and vector storage.
- **Google Generative AI**: Used to generate answers to questions.
- **FAISS**: For vector searches.
- **dotenv**: For environment variables, such as the Google API key.

## Conclusion:
This application uses powerful AI and vectorization technologies to extract content from PDF files, break it into meaningful chunks, and then generate answers to user questions based on that content.
