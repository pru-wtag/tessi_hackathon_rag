import os
from typing import List, Union, Generator, Iterator

from pydantic import BaseModel
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
import asyncio


class Pipeline:

    class Valves(BaseModel):
        LLAMAINDEX_OLLAMA_BASE_URL: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str

    def __init__(self):
        self.documents = None
        self.index = None

        self.valves = self.Valves(
            LLAMAINDEX_OLLAMA_BASE_URL=os.getenv("LLAMAINDEX_OLLAMA_BASE_URL", "http://localhost:11434"),
            LLAMAINDEX_MODEL_NAME=os.getenv("LLAMAINDEX_MODEL_NAME", "llama3.1:8b"),
            LLAMAINDEX_EMBEDDING_MODEL_NAME=os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "jina/jina-embeddings-v2-base-en:latest"),
        )

    async def on_startup(self):
        Settings.embed_model = OllamaEmbedding(
            model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )
        Settings.llm = Ollama(
            model=self.valves.LLAMAINDEX_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )

        self.documents = SimpleDirectoryReader("dataset").load_data()
        self.index = VectorStoreIndex.from_documents(self.documents)

    async def on_shutdown(self):
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        query_engine = self.index.as_query_engine(streaming=True)
        response = query_engine.query(user_message)
        return response.response_gen

async def main():
    # Instantiate the pipeline
    pipeline = Pipeline()

    # Simulate the server startup process
    await pipeline.on_startup()

    # Define a sample question
    sample_question = "What are the benefits of using renewable energy sources?"

    # Simulate a user message
    user_message = sample_question
    model_id = "sample_model"  # This can be a placeholder or an actual model ID
    messages = [{"role": "user", "content": user_message}]
    body = {}  # Additional context or parameters if needed

    # Run the pipeline with the sample question
    response_gen = pipeline.pipe(user_message, model_id, messages, body)

    # Collect and print the response
    response_text = ''.join(response_gen)
    print("\nResponse:")
    print(response_text)

    # Simulate the server shutdown process
    await pipeline.on_shutdown()

if __name__ == "__main__":
    asyncio.run(main())