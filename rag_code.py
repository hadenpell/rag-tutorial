import nest_asyncio
import os
from dotenv import load_dotenv

from llama_index.llms.openai import OpenAI
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex

from IPython.display import display, Markdown

from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.workflow import Context

nest_asyncio.apply()

# Get variables from .env
load_dotenv()

def create_tool():
    # Create index
    index = LlamaCloudIndex(
        name=os.getenv("INDEX_NAME"), 
        project_name="Default",
        organization_id=os.getenv("ORG_ID"),
        api_key=os.getenv("LLAMA_API_KEY")
    )

    # Create query engine from index
    llama_cloud_query_engine = index.as_query_engine()

    # Create tool from query engine
    llama_cloud_tool = QueryEngineTool.from_defaults(
        query_engine=llama_cloud_query_engine,
        description=(
            f"Useful for answering semantic questions about certain states in the US."
        ),
        name="llama_cloud_tool"
    )

    return llama_cloud_tool

def create_agent():
    # Create tool
    tool = create_tool()
    # Create agent from tool
    agent = FunctionAgent(
        name="state-guide-usa",
        description= f"Useful for answering semantic questions about certain states in the US.",
        tools=[tool],
        llm=OpenAI(model="gpt-4o-mini"),
        system_prompt="You are useful for answering semantic questions about certain states in the US.",
    )
    # Add context
    ctx = Context(agent)
    return agent, ctx