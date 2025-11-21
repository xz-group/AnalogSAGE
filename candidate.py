import os
import pickle
import numpy as np
import evalTask


class TopologyDatabase:
    def __init__(self):
        self.embeddings = []
        self.texts = []

    def load_pkls(self, folder: str):
        for file in os.listdir(folder):
            if file.endswith(".pkl"):
                path = os.path.join(folder, file)
                with open(path, "rb") as f:
                    data = pickle.load(f)
                    self.embeddings.append(data)
                    self.texts.append(file)
        self.embeddings = np.array(self.embeddings)
        print(f" {len(self.texts)} ")

    def cosine_similarity(self, a, b):
        a_norm = a / (np.linalg.norm(a) + 1e-10)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
        return np.dot(b_norm, a_norm)

    def query(self, q_embedding, threshold=0.8, top_k=10):
        q_embedding = np.array(q_embedding)
        sims = self.cosine_similarity(q_embedding, self.embeddings)
        top_idx = np.argsort(-sims)[:top_k]
        results = []
        for idx in top_idx:
            results.append({"text": self.texts[idx], "similarity": float(sims[idx])})
        # print(str(results))
        return results


from openai import OpenAI

client = OpenAI(
    api_key=""
)
import re


def extract_code(text):
    pattern = r"```(?:\w+)?\s*([\s\S]*?)```"
    matches = re.findall(pattern, text)
    return [m.strip() for m in matches]


def embedding(myText):
    response = client.embeddings.create(input=myText, model="text-embedding-3-large")

    # print(response.data[0].embedding)
    return response.data[0].embedding


def askLLM(prompt, mymodel):

    if mymodel == "gpt-4o" or mymodel == "gpt-5":
        client = OpenAI(
            api_key=""
        )
    else:
        client = OpenAI(
            api_key="",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    response = client.chat.completions.create(
        model=mymodel,
        messages=[
            {
                "role": "system",
                "content": "You are an expert in analog circuit design.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def getNetlist(id):
    answer = ""
    i = id
    with open(
        f"",
        "r",
    ) as f:
        netlist = f.read()
    answer += f"Netlist id: {i}\n"
    answer += netlist + "\n\n"
    return answer


def getCandidates(query_text, spec, queryID):
    database = TopologyDatabase()
    database.load_pkls("")
    second_query = f"""
Topology description: 
{query_text}
Please write a prompt to search for this topology design in a RAG system and only output the topology description prompt text directly without any additional explanation.
Sample output: 
The circuit is a two-stage operational amplifier. The first stage consists of an NMOS differential pair with PMOS diode-connected loads, while the second stage is a common-source amplifier with a self-cascoded PMOS driver and an NMOS current-source load to achieve high gain.
"""
    answer = askLLM(second_query, "gemini-2.5-flash")
    query_embedding = embedding(answer)
    results = database.query(query_embedding)
    mynewcandidate = ""
    for res in results:
        text = res["text"]
        # similarity = res["similarity"]
        id = int(text.split("_")[0][4:])
        mynewcandidate += getNetlist(id)
    third_query = f"""
I have the spec requirements of an op-amp (under skywater PDK, SKY130 process node) as follows:
{spec}
And I have a topology design description that could meet the spec requirements as follows:
{query_text}
I also have some candidate netlists that may meet the spec requirements as follows:
{mynewcandidate}
Please help me to select the best candidate netlist that meets the spec requirements.
First compare each candidate with the topology design description and eliminate those that do not match the topology design and finally output the topology candidate netlist id directly sorted based on the possibility that they meet the spec requirements.
Example:
```python
mycandidate = [35,17] # 35 is the best candidate, 17 is the second best candidate, etc. Give the best 5 candidates.
```
Please do not modify the variable name "mycandidate" and put it in a python code block.
"""
    print("Third query:")
    print(third_query)
    answer = askLLM(third_query, "gemini-2.5-flash")
    print("Final candidate selection result:")
    print(answer)
    env={}
    try:
        candidates = extract_code(answer)
        
        exec(candidates[0],env)
        totalcandidate = env["mycandidate"]
        # print("candidates:")
        # print(totalcandidate)
        # print(mycandidate[0])
        # exec(mycandi)
        eval_result = evalTask.evalTask(queryID, totalcandidate)
    except Exception as e:

        print(f"Error: {e}")
        totalcandidate = []
        eval_result = 0
    print(f"Eval: {eval_result}")
    return (mynewcandidate,totalcandidate)

