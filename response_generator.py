import os
import openai
import json


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

    def generate_response_baygata(self, query, context_docs):
        prompt = {
            "Instruction": "The user has asked for images according to the query. Choose from the Available Images and return their IDs.",
            "User Query": query,
            "Available Images": context_docs,
            "Output": """Return a JSON array of the ids for the relevant images like below. Don't say anything else.

{"ids": ["id1", "id2", "id3", ...]}""",
        }

        gpt_messages = [
            {"role": "user", "content": json.dumps(prompt)},
        ]

        response = openai.chat.completions.create(model=os.getenv("OPENAI_MODEL_NAME"), messages=gpt_messages)
        content = response.choices[0].message.content
        if type(content) == str:
            if content.startswith("```json"):
                content = content.replace("```json", "", 1)
            if content.endswith("```"):
                content = content[:-3]
            return json.loads(content)
        return content
