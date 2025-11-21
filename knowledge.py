import queryPinecone
from openai import OpenAI


def askLLM_with_history(prompt, history):
    client = OpenAI(
            api_key="",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    messages = history + [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=messages,
    )
    return response.choices[0].message.content


import re


def extract_code(text):
    pattern = r"```(?:\w+)?\s*([\s\S]*?)```"
    matches = re.findall(pattern, text)
    return [m.strip() for m in matches]


def getKnowledge(query_text, mytop_k=5):
    history = [
        {
            "role": "system",
            "content": "You are an expert in analog circuit design.",
        },
    ]

    prompt = f"""
You are a knowledge retrieval agent for analog circuit design. Given a query, you will first convert the knowledge query into several RAG query and then, you will summary the retrieval documents.
Here is the query: {query_text}
The knowledge base contains many published papers about analog circuit design but do not have the specific domain knowledge for a certain PDK.
First analysis the query and revise it into several sub-queries to retrieve relevant knowledge. 
You should return the queries in the following python format:
```python
queries = ["query1", "query2", ...]
```
Do not change the variable name "queries". At most 5 elements in queries.
"""
    query_result = askLLM_with_history(prompt, history)
    print("Query decomposition result:")
    print(query_result)
    history = history + [{"role": "assistant", "content": query_result}]
    code_snippets = extract_code(query_result)
    print("Extracted code snippets:")
    print(code_snippets[0])
    namespace = {}
    exec(code_snippets[0],namespace)
    queries = namespace["queries"]
    print("Decomposed queries:")
    print(queries)

    results = queryPinecone.queryPinecone(queries, mytop_k)
    print("Retrieved results:")
    for res in results:
        print(res)
        print("-----")
    new_prompt = "Now I have retrieved the following relevant knowledge documents:\n"
    for res in results:
        new_prompt += res["text"] + "\n"
    new_prompt += f"Please summarize the relevant knowledge to help me answer the original query: {query_text}. Provide detailed and technical information. Do not give unrelated information about the query."
    summary = askLLM_with_history(new_prompt, history)
    print("Knowledge summary:")
    print(summary)
    return summary


if __name__ == "__main__":
    query_text = "I want to design an op-amp with high gain and high bandwidth. How can I achieve this?"
    getKnowledge(query_text, mytop_k=5)
