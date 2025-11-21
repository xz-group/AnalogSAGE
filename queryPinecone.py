from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

def queryPinecone(query_text_list,mytop_k=10):
    pinecone_client = Pinecone(
                api_key = ""
            )
    index = pinecone_client.Index(
                name ='research-papers-pa',
            )

    embedding_model = 'text-embedding-3-large'
    openai_embedding_model = OpenAIEmbeddings(
            model = embedding_model,
            openai_api_key = ""
        )
    ans = []
    for i in range(len(query_text_list)):
        query_text1 = query_text_list[i]
        # print(f"Query {i+1}: {query_text1}")
        query_embedding = openai_embedding_model.embed_query(query_text1)
        retrieved_results = index.query(
                        vector = query_embedding,
                        top_k = mytop_k,
                        include_values = False,
                        include_metadata = True,
                        namespace="ISSCC"
                    )['matches']

        for i, result in enumerate(retrieved_results):
            print(result['metadata'])
            print("-----")
            ans.append(result['metadata'])
    return ans