# Call LLM model
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",api_key="AIzaSyCxpS4DpYHn8vmDVk04uYDeyIODFpz07hU")
#from langchain_ollama import OllamaLLM
#llm=OllamaLLM(model="llama3")

# create an embedding object
from langchain_huggingface import HuggingFaceEmbeddings
embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


# Create vector database
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader

def chunk_data(data, chunk_size=30):
    # Chunk the documents into smaller pieces
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunks.append(data[i:i+chunk_size])
    return chunks

def create_vdb():
    # Load the CSV file
    loader = CSVLoader(file_path='amazon_data.csv', source_column="question")
    data = loader.load()
    # Chunk the data to avoid overwhelming the vector database
    chunked_data = chunk_data(data)
    # Create the vector database using Chroma for each chunk
    print("Creating vdb")
    for idx, chunk in enumerate(chunked_data):
        Chroma.from_documents(chunk, embedding=embed, persist_directory=f"amazon_vdb_chunk_{idx}")
    
    print("Done creating vdb")
    


# Create a function to get the answer/chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

def get_answer(query):
    
    # Define the system prompt for the model
    system_prompt = (
        "Use the context to answer the question. "
        "If you know the answer, keep the answer concise (maximum 10 sentences)."
        "Give the answer in stepwise if possible."
        "Context: {context}"
    )
    # Configure prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    #lets loop over all the different vdb
    for i in range(4):
        # Load the vector database
        vdb = Chroma(persist_directory=f"amazon_vdb_chunk_{i}", embedding_function=embed)
        # Configure retriever to retrieve metadata
        retriever = vdb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'score_threshold': 0.3}
            )    
        # Chain to combine retrieval and response generation
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        result=rag_chain.invoke({"input": query})
        #if answer has relevant content i.e context is present, skip the loop
        if result["context"]:
            return result["answer"]
    
    return print("Please go to Amazon website for more details because your question is not a part of Amazon FAQ")
            

if __name__ == "__main__":
    #create_vdb()
    query = "How to use a coupon?"
    print(get_answer(query))