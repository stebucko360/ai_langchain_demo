from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

llm = Ollama(model="llama2")
## ingests document to pass to vector store
embeddings = OllamaEmbeddings()
output_parser = StrOutputParser()
## web based loader to scrape web page
loader = PyPDFLoader("./sample.pdf")
##loads the given doc and splits into chunks
pages = loader.load_and_split()
## Build index for vector store from ingested document
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(pages)
vector = FAISS.from_documents(documents, embeddings)

##prompt for query
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

##retrival chain for pulling in relevant doc
retriever = vector.as_retriever()
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)


chain = prompt | llm | output_parser

response = retrieval_chain.invoke({"input": "summarise this document in one sentence"})
print(response["answer"])

# message = chain.invoke({"input": "So llama2 is just considered a package and is installed globally?"})

# print("REPLY:" + message)