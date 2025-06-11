import streamlit as st
from dotenv import load_dotenv
import nest_asyncio, asyncio
from rag_code import create_agent
import gc

load_dotenv()

# Apply nest_asyncio to allow running asyncio in Streamlit
nest_asyncio.apply()

# Initialize session state 
if "messages" not in st.session_state:
    st.session_state.messages = []

# # Initialize agent and context
# if "agent" not in st.session_state:
#     st.session_state.agent, st.session_state.ctx = create_agent()

def reset_chat():
    st.session_state.messages = []
    st.session_state.ctx = None
    gc.collect()

# Set page configuration
st.set_page_config(
    page_title="US State Guide",
    page_icon="🏙️",
    layout="wide",
)

# Set app title and description 
st.markdown(
    """
    <h1 style='text-align: center;'>
         <span style='color:#FF4B4B;'>US State Guide 🏙️</span>
    </h1>
    <h3 style='text-align: center;'>Powered by OpenAI & LlamaCloud</h3>
    """,
    unsafe_allow_html=True
)

st.markdown("""
**Ask questions about New York, California, Florida, Washington DC, and Texas.**
            
Pulls facts from Wikipedia pages of U.S. states
""")
st.markdown("""
**Examples:**
- What is the population of New York?
- What is there to do for fun in Florida?
- How do people in Chicago get around? """)

# Run workflow and process query, return result
async def run_workflow(query):
    try:
        agent, ctx = create_agent()
        result = await agent.run(user_msg=query, ctx=ctx)
        return str(result)
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return f"I'm sorry, I encountered an error: {str(e)}"

# Add sidebar for chat history
with st.sidebar:
    st.title("Chat History")
    for message in st.session_state.messages:
        role = "👤" if message["role"] == "user" else "🤖"
        st.text(f"{role} {message['content']}")

# Display at most only the 2 most recent questions and answers
if len(st.session_state.messages) >= 2:
    for message in st.session_state.messages[-2:]: 
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Get user input
user_query = st.chat_input("Ask a question about a US state!")

if user_query:
    # Add message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Display message
    with st.chat_message("user"):
        st.write(user_query)

    # Display response with a "thinking" message while processsing
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = asyncio.run(run_workflow(user_query))
                st.write(response)
                # Add response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Add "Clear" button for clearing chat
col1, col2 = st.columns([6, 1])
with col2:
    st.button("Clear Chat", on_click=reset_chat)

