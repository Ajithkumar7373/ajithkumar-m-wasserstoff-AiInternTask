from tabulate import tabulate  # For table formatting
from langchain_community.document_loaders import PyPDFLoader
from PIL import Image
import pytesseract
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from keybert import KeyBERT  # For theme extraction

class Rag:
    def __init__(self):
        self.chain = None
        self.kw_model = KeyBERT()

    def process_documents(self, pdf_paths, image_paths, persist_dir="chroma_db"):
        """
        Reads PDFs and scanned images, extracts text, identifies themes,
        and stores them in ChromaDB.
        """
        documents = []

        # Load PDFs
        for doc_num, pdf_path in enumerate(pdf_paths, start=1):
            loader = PyPDFLoader(pdf_path)
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["document_id"] = doc_num  # Assign doc ID
                doc.metadata["themes"] = self.extract_themes(doc.page_content)
                documents.append(doc)

        # Extract text from scanned images
        for img_num, img_path in enumerate(image_paths, start=len(pdf_paths) + 1):
            text = pytesseract.image_to_string(Image.open(img_path))
            themes = self.extract_themes(text)
            image_doc = Document(
                page_content=text,
                metadata={"document_id": img_num, "document_type": "image", "themes": themes}
            )
            documents.append(image_doc)

        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        docs = splitter.split_documents(documents)

        # Create embeddings
        embedding = OllamaEmbeddings(model="nomic-embed-text")
        vectorstore = Chroma.from_documents(docs, embedding=embedding)

        # Setup retriever
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        # Setup language model
        llm = ChatOllama(model="gemma:2b", temperature=0)

        # Define system prompt
        system_prompt = ("You are an assistant for question-answering tasks. "
                         "Use the following pieces of retrieved context to answer the question. "
                         "If you don't know the answer, say 'I don't know'. Use max 3 sentences.\n\n{context}")

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}")
            ]
        )

        # Create RAG pipeline
        question_answerer_chain = create_stuff_documents_chain(llm, prompt)
        self.chain = create_retrieval_chain(retriever, question_answerer_chain)

        # Print themes in table format
        self.display_theme_table(docs)

    def extract_themes(self, text: str) -> list:
        """
        Extracts key themes or topics from text using KeyBERT.
        """
        themes = self.kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=5)
        return [t[0] for t in themes]  # Return only theme names

    def display_theme_table(self, docs):
        """
        Displays document themes in a structured table format.
        """
        table_data = []
        for doc in docs:
            doc_id = doc.metadata.get("document_id", "N/A")
            page_num = doc.metadata.get("page", "N/A")
            themes = ", ".join(doc.metadata.get("themes", []))

            table_data.append([doc_id, page_num, themes])

        print(tabulate(table_data, headers=["Document ID", "Page Number", "Themes"], tablefmt="grid"))

    def ask(self, query: str) -> str:
        if not self.chain:
            return "Error: No document data is available. Please run `process_documents` first."

        response = self.chain.invoke({"input": query})
        return response.get("answer", "I don't know.")