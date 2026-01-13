from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


def get_rag_chain(vectorstore_path="faiss_index"):
    # Load embeddings & vector store
    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.load_local(
        vectorstore_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question using ONLY the context below.
        If the answer is not in the context, say "I don't know".

        Context:
        {context}

        Question:
        {question}
        """
    )

    def rag_pipeline(question: str):
        docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)

        chain = prompt | llm
        response = chain.invoke(
            {"context": context, "question": question}
        )

        return response.content

    return rag_pipeline