from openai import OpenAI
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


def compress_feedback(history,special_prompt =""):
    
    prompt=history + "\n You were unsuccessful in completing the design task in the iterations. The above is the whole chat history of the design process. Please compress the design and the performance result among iterations as feedback to the previous steps to make it generate a better design. Use complete sentences. Summarize the key points and avoid redundancy. Be concise and to the point. "
    prompt += special_prompt
    feedback = askLLM(prompt, "gemini-2.5-flash")
    return feedback
