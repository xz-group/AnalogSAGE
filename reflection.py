from openai import OpenAI
def askLLM(prompt, mymodel):

    if mymodel == "gpt-4o" or mymodel == "gpt-5":
        client = OpenAI(
            api_key="ABCD"
        )
    else:
        client = OpenAI(
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


def get_reflection(history):
    prompt=history + "\n You were unsuccessful in completing the design task in this trial. The above is the entire chat history of the design process in this agent. In a few sentences, diagnose a possible reason for failure that aims to mitigate the same failure. Use complete sentences. Note that the next trial will start from topology selection with this reflection. Do not mention detail sizing as the next trial will have a new sizing process under another topology. Please focus on the topology level and include the current topology because the next trial will not know the topology used in this trial. "
    feedback = askLLM(prompt, "gemini-2.5-flash")
    return feedback
