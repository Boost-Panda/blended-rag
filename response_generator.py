import os
import openai


class ResponseGenerator:
    def __init__(self, openai_api_key):
        openai.api_key = openai_api_key

    def generate_response(self, query, context_docs):
        context = " ".join(context_docs)

        prompt = """
        You are a helpful chat assistant that answers questions about various topics. The context is provided below, followed by a question that you need to answer. Please provide a concise and informative response to the question. If you cant find the answer, you can say "I don't know".
        """

        gpt_messages = [
            {"role": "system", "content": prompt},
            {"role": "system", "content": context},
            {"role": "user", "content": query},
        ]
        
        response = openai.chat.completions.create(
            model=os.getenv("OPENAI_MODEL_NAME"),
            messages=gpt_messages,
        )
        return response.choices[0].message.content
