This project is a web-based application that allows users to upload Excel files and then ask questions about the content in those Excel sheets. The AI model processes the data and responds with relevant, context-specific answers.

Features
File Upload

Users can upload Excel files (both .xlsx and .xls). The app extracts and processes the text from these sheets.
AI-Powered Q&A

Uses a language model (OpenAI or Hugging Face embeddings + ChatOpenAI) for answering questions about the uploaded Excel data.
Provides detailed answers; if the context is missing, it indicates so.
User-Friendly UI

Built with Streamlit for quick and intuitive user interaction.
Convenient text input for questions and sidebars for file uploads.
Deployment

Deployed via Streamlit Sharing / any public platform of your choice.
A live link is provided so users can easily interact with the application.
How It Works
Upload Excel Files
Drag and drop or select multiple Excel files.
Process Files
The content from these files is split into text chunks, embedded, and stored in a vector store (FAISS).
Ask Questions
Enter your question in the text input box.
The app retrieves the most relevant chunks from the vector store and passes them to the AI model.
The model then generates a precise answer based on the context provided by those chunks.
Tech Stack
Language Model: OpenAI ChatOpenAI / Hugging Face Sentence Transformers for embeddings
Vector Store: FAISS (via langchain_community)
UI: Streamlit
Environment Management: python-dotenv for handling environment variables (OpenAI API key).
