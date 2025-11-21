import os
import pickle
import numpy as np
import knowledge

# {"1":(id,"myinsight")}
myinsight = {}


def getMyknowledgeSpec(id):
    startI = 1
    if id ==1:
        return ""
    myprompt = "I have designed these topologies with spec requirements."

    while startI < id:
        if str(startI) not in myinsight:
            startI += 1
            continue
        currentTopologyId = myinsight[str(startI)][0]
        insight = myinsight[str(startI)][1]
        # Your requirement path here
        requirement = open(f"TASK").read()
        topology = open(
            # Your topology description path here
            f"TOPOLOGY"
        ).read()
        myprompt = (
            myprompt
            + f"""
Query {startI} Requirement:
{requirement}
Query {startI} Topology description:
{topology}
Query {startI} Insight description:
{insight}
"""
        )
        startI = startI + 1

    return myprompt

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

    def query(self, q_embedding, top_k=10):
        q_embedding = np.array(q_embedding)
        sims = self.cosine_similarity(q_embedding, self.embeddings)
        top_idx = np.argsort(-sims)[:top_k]
        results = []
        for idx in top_idx:
            results.append({"text": self.texts[idx], "similarity": float(sims[idx])})
        return results


from openai import OpenAI

def embedding(myText):
    client = OpenAI(
        # Your API key here
        api_key="ABCD"

)
    response = client.embeddings.create(input=myText, model="text-embedding-3-large")

    # print(response.data[0].embedding)
    return response.data[0].embedding


def askLLM(prompt, mymodel):

    if mymodel == "gpt-4o" or mymodel == "gpt-5":
        client = OpenAI(
            # Your API key here
        api_key="ABCD"
        )
    else:
        client = OpenAI(
             # Your API key here
            api_key="ABCD",
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
import re
def extract_code(text):
    pattern = r"```(?:\w+)?\s*([\s\S]*?)```"
    matches = re.findall(pattern, text)
    return [m.strip() for m in matches]
from datetime import datetime


if __name__ == "__main__":
    database = TopologyDatabase()
    #Dataset path here
    database.load_pkls("")

    modellist = ["gemini-2.5-flash"]
    now_str = datetime.now().strftime("%m%d_%H%M%S")

    result_test_path = f"testResult _{now_str}.txt"
    result_test_file = open(result_test_path, "w")
    result_test_file.write("Test results\n")
    for model in modellist:
        for query in [1,2,3,4,5,6,7,8,9,10]:
            result_test_file.write("Current insight:\n")
            result_test_file.write(str(myinsight))
            result_test_file.write("\n")
            for testK in [1,2,3,4,5]:
                myTestK_is_success=False
                iteration = 0
                my_reflection = []
                while not myTestK_is_success: 
                    now_str = datetime.now().strftime("%m%d_%H%M%S")
                    result_test_file.write(f"Model {model} Query {query} Test {testK}\n")
                    print(f"Query {query} Test {testK} Model {model}")
                    reflection_prompt = ""
                    if len(my_reflection) > 0:
                        reflection_prompt = "\n Here are the reflections summarized from the previous trial:" + str(my_reflection) + "\n"
                        

                    with open(f"", "r") as f:
                        spec = f.read()
                    testPrompt = getMyknowledgeSpec(query)
                    insights = ""
                    zero_query = insights + reflection_prompt+ f"""
    I want to design an operational amplifier using the skywater PDK (SKY130 process node) that meets the following specifications:
    {spec}
    I have an knowledge agent that can help me find relevant knowledge documents about analog circuit design from a knowledge base.
    Please help me to write a query to ask the knowledge agent to get relevant knowledge documents to help me design the op-amp.
    First analyze the specifications and then output the query text in a code block.
    Example
    ```text
    I want to design an op-amp with high gain and high bandwidth. How can I achieve this?
    ```
    """
                    knowledge_query = askLLM(zero_query, model)
                    knowledge_query = extract_code(knowledge_query)[0]
                    print("Knowledge query: ", knowledge_query)
                    knowledge_docs = knowledge.getKnowledge(knowledge_query, mytop_k=5)
                    print("Knowledge docs: ", knowledge_docs)
                    print("Knowledge text: ", knowledge_docs)

                    first_query = f"""
    Please describe a possible topology design of a op-amp (under skywater PDK, SKY130 process node) that meets the following specifications:
    {spec}
    I have the following relevant knowledge documents to help me design the op-amp:
    {knowledge_docs}
    """
                    first_query = insights + reflection_prompt + "\n" + first_query

                    history =first_query
                    answer = askLLM(first_query, model)
                    history += ("\n" + answer)
                    import candidate
                    mynewcandidate = candidate.getCandidates(answer, spec,query)
                    print(str(mynewcandidate))
                    print("-----")
                    print(str(mynewcandidate))
                    result_test_file.write(f"results:\n{str(mynewcandidate)}\n")
                    import evalTask
                    queryID = query
                    totalcandidate = mynewcandidate[1]
                    eval_result = evalTask.evalTask(queryID, totalcandidate)
                    print(f"Eval: {eval_result}")
                    result_test_file.write(f"Eval: {eval_result}\n")
                    iteration += 1
                    result_test_file.write(f"Iteration: {iteration}\n")
                    print(f"Iteration: {iteration}\n")
                    if eval_result >= 1:
                        myTestK_is_success=True
                        import insight
                        myNetlist = evalTask.evalTaskBestFit(queryID, totalcandidate)
                        my_new_insight = insight.get_insight(history)
                        myinsight[str(queryID)] = (myNetlist,my_new_insight)
                        print("New insight: ", my_new_insight)
                        # log_file.write("New insight: \n" + my_new_insight + "\n")
                        result_test_file.flush()
                        break
                    else:
                        if iteration >=3:
                            print("Reached max iterations, stopping...")
                            result_test_file.write("Reached max iterations, stopping...\n")
                            result_test_file.flush()
                            break
                        print("Retrying...")
                        import reflection
                        import simulation
                        history += ("\n" + "Now I get this design:")
                        history_fixed = history
                        for netlistId in totalcandidate:
                            mynetlist = open(f"").read()
                            performance = simulation.doSimulate({},queryID,netlistId) 
                            history =history_fixed + ("\n" + "The sized performance is")
                            history +=str(performance)
                            my_reflection_new = reflection.get_reflection(history)
                            my_reflection.append(my_reflection_new)
                            print("Reflection: ", my_reflection_new)
                            result_test_file.write("Reflection: \n" + my_reflection_new + "\n")
                    result_test_file.flush()
