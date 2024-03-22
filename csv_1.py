from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders.csv_loader import CSVLoader


llm = Ollama(model="llama2")
## ingests document to pass to vector store
print("Creating embeddings for Vector ingest")
embeddings = OllamaEmbeddings()
output_parser = StrOutputParser()
## web based loader to read the csv file
print("Loading CSV")
loader = CSVLoader(file_path='./country.csv')
##loads the given doc and splits into chunks
print("Splitting CSV")
pages = loader.load_and_split()
## Build index for vector store from ingested document
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(pages)
print("Vector initalising")
vector = FAISS.from_documents(documents, embeddings)

##prompt for query
print("Reading prompt...")
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

##retrival chain for pulling in relevant doc
print("Retrieval Chain starting up")
retriever = vector.as_retriever()
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

print("Lets gooooo")
chain = prompt | llm | output_parser

response = retrieval_chain.invoke({"input": "Which countries have the letter Z?"})
print(response["answer"])

# message = chain.invoke({"input": "So llama2 is just considered a package and is installed globally?"})

# print("REPLY:" + message)