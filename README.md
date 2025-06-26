# US State Guide 🇺🇸: Using RAG and LlamaCloud to Build a Custom Chatbot

This project creates a custom chatbot that can query a LlamaCloud index for RAG-based retrieval.
In this example, we'll use PDFs of Wikipedia pages of US states to train our model.

We use:
* LlamaIndex for orchestrating the RAG app.
* Streamlit to build the UI.

## Installation and setup

#### Setup OpenAI
Get an API key from OpenAI and set it in the .env file as follows:
```
OPENAI_API_KEY = YOUR_OPENAI_API_KEY
```

#### Setup LlamaCloud 
Get an API key from LlamaCloud and set it in the .env file as follows:
```
LLAMA_API_KEY = YOUR_LLAMA_API_KEY
```

<b>Install Dependencies</b>: Ensure you have Python 3.10 or later installed.

```
pip install -r requirements.txt
```
 
<b>Run the app:</b>

Run the app by running the following command:
```
streamlit run app.py
```