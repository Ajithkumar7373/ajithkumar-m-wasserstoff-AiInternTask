from fastapi import FastAPI, UploadFile, File
import shutil
import os

from pydantic import BaseModel

from load import Rag
from config import set_cors
rag=Rag()
app = FastAPI()
set_cors(app)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/upload/")
async def upload_files(pdf: UploadFile = File(None), image: UploadFile = File(None)):
    """
    Upload PDFs and images, save them to the server, and process them using RAG.
    """
    pdf_path = None
    image_path = None

    # Save PDF
    if pdf:
        pdf_path = os.path.join(UPLOAD_FOLDER, pdf.filename)
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(pdf.file, f)

    # Save Image
    if image:
        image_path = os.path.join(UPLOAD_FOLDER, image.filename)
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

    # Process stored documents
    pdf_paths = [pdf_path] if pdf_path else []
    image_paths = [image_path] if image_path else []

    Rag.process_documents(pdf_paths, image_paths)

    return {"message": "Files uploaded and processed successfully!"}

class Query(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

@app.get("/themes/")
def get_themes():
    global rag  # Ensure we're using the global rag object

    if not rag:
        return {"error": "No document data is available. Please upload and process documents first."}

    try:
        return {"themes": rag.display_theme_table}
    except Exception as e:
        return {"error": str(e)}




@app.post("/ask",response_model=AnswerResponse)


async def ask_question(query: Query):
    answer =rag.ask(query.question)
    return AnswerResponse(answer=answer)
