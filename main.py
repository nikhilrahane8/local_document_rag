from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
import sys
import os


sys.stdout.reconfigure(encoding='utf-8')
pdf_path="Rules.pdf"
loader=PyPDFLoader(pdf_path)
document=loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
chunks=text_splitter.split_documents(document)



embeddings=HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")
llm = OllamaLLM(model="qwen2.5:0.5b", temperature=0)


db=Chroma.from_documents(documents=chunks,embedding=embeddings)

while True:
    query=input("You :")
    if query=="quit":
        print("vbot: Goodbye!")
        break

    matching_docs=db.similarity_search(query,k=3)

    context="\n\n---\n\n".join([doc.page_content for doc in matching_docs])
    prompt=f"""You are a strict, helpful AI assistant. Read the CONTEXT below and answer the QUESTION based ONLY on the CONTEXT. If the answer is not in the context, say"I cannot find this in the document."

### CONTEXT:
{context}

### QUESTION:
{query}

### ANSWER:"""

    response=llm.invoke(prompt)
    print(f"vbot: {response}")
