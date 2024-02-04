import argparse
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
# from langchain.chat_models import ChatOpenAI
# from openai import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
# ChatOpenAI()

def get_db():
    load_dotenv()
    dbs = []
    for category in ['research','sales', 'decision']:
        embedding_function = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        db = Chroma(persist_directory=CHROMA_PATH+"\\"+category, embedding_function=embedding_function)
        dbs.append(db)
    return dbs

def  query_db(query_text, category):
    load_dotenv()

    # Create CLI.
    # parser = argparse.ArgumentParser()
    # parser.add_argument("query_text", type=str, help="The query text.")
    # args = parser.parse_args()
    # query_text = args.query_text
    

    # Prepare the DB.
    embedding_function = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    db = Chroma(persist_directory=CHROMA_PATH+"\\"+category, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    # print("---------------------------------------------------------------------\n", results)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2", model_kwargs={"temperature":0.5, "max_length":512})
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    # print(formatted_response)


    # print("//////////////////")
    # print(response_text)
    return response_text


if __name__ == "__main__":
    query_db()