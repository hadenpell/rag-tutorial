import nest_asyncio
import os
from dotenv import load_dotenv

from llama_index.llms.openai import OpenAI
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex

from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer

nest_asyncio.apply()

# Get variables from .env
load_dotenv()

def create_index():
    # Create index
    index = LlamaCloudIndex(
        name=os.getenv("INDEX_NAME"), 
        project_name="Default",
        organization_id=os.getenv("ORG_ID"),
        api_key=os.getenv("LLAMA_API_KEY")
    )

    return index

def create_chat_engine():
    index = create_index()
    retriever = index.as_retriever()
    memory = ChatMemoryBuffer.from_defaults(token_limit=2000)
    
    chat_engine = ContextChatEngine.from_defaults(
        retriever=retriever,
        memory=memory,
        llm=OpenAI(model="gpt-4o"),
    )
    return chat_engine