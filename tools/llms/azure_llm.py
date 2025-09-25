import openai
from tools.llms.base_llm import BaseLLM
import os

class AzureLLM(BaseLLM):
    def __init__(self, deployment_name=None, max_completion_tokens=512):
        openai.api_type = "azure"
        openai.api_base = os.environ.get("AZURE_OPENAI_ENDPOINT")  # required
        openai.api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        openai.api_key = os.environ["AZURE_OPENAI_API_KEY"]

        # Use provided deployment_name if given
        self.deployment_name = deployment_name or os.environ["DEPLOYMENT_NAME"]
        self.max_completion_tokens = max_completion_tokens

    def generate(self, query: str, context: list, system_prompt: str = None):
        """
        query: user's question
        context: list of retrieved passages
        system_prompt: optional custom system prompt for different agents
        """
        context_text = "\n\n---\n\n".join(context)

        # Default system prompt
        system_prompt = system_prompt or "You are an assistant that answers questions ONLY using the provided context."

        user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer concisely."

        response = openai.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=self.max_completion_tokens
        )
        print("Context", context)
        return response.choices[0].message.content.strip()
