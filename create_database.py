from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores.chroma import Chroma

import os
import shutil

CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():
    generate_data_store()


def generate_data_store():
    for category in ['research','sales', 'decision']:
        documents = load_documents(category)
        chunks = split_text(documents)
        save_to_chroma(chunks, category)


def load_documents(category):
    loader = DirectoryLoader(DATA_PATH + "\\" + category , glob="*.md")
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[1]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document], category: str):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH + "\\" + category):
        shutil.rmtree(CHROMA_PATH + "\\" + category)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl"), persist_directory=CHROMA_PATH + "\\" + category
    )
    db.persist()
    #print(f"Saved {len(chunks)} chunks to {CHROMA_PATH + "\\" + type}.")


if __name__ == "__main__":
    main()