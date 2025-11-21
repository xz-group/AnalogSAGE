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


def get_insight(history):
    prompt=history + "\n You have successfully completed the design task. The above is the entire chat history of the design process. In a few sentences, give an insight observation you find in this design process which could be used to guide another op-amp design under a different specification requirement. Use complete sentences."
    feedback = askLLM(prompt, "gemini-2.5-flash")
    return feedback
