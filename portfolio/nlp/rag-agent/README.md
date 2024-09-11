# Local Psychotherapy RAG and Web Agent

Psy AI is an agent that leverages local Ollama Llama 3.1 8b model together with RAG and web capabilites to provide insights based on user input. Its prompting is optimized to psychology, psychotherapy, and psychiatry to help user in most professional yet caring manner.

## How It Works

### Local Ollama Llama3.1 8b Agent (`agent.py`)
The Psy AI assistant is built using the locally hosted Ollama Llama 3.1 8b model, which serves as the foundational language model for the system. Tests were conducted using Nvidia Geforce RTX 4090, which provided low latency during model interactions. If you wish, you can substitute default model to any desired one whether based on API or local.

### LangGraph Architecture

The internal logic of the agent is based on LangGraph framework. This graph controls the flow of actions, retrievals, and state management. The agent dynamically updates its knowledge and state as it interacts with the user.

![LangGraph Structure](images/graph.png)

### LlamaIndex Retrieval-Augmented Generation
The agent uses LlamaIndex, which powers the RAG process. The assistant utilizes RAG to intelligently retrieve relevant information from local knowledge base documents, ensuring that responses are grounded in accurate, contextually relevant data.

The assistant interacts with a vector database, such as ChromaDB, to access indexed knowledge. This enables the system to answer complex queries by fetching related documents and using them in conjunction with the language model to generate responses. This approach enhances the assistant’s ability to provide precise and well-informed answers based on external knowledge beyond its pre-existing training.

### Tavily
In addition to the RAG system, the assistant is equipped with Tavily Web, allowing it to fetch real-time data from the web when necessary. This integration ensures that the assistant can go beyond its internal knowledge base and offer up-to-date, web-sourced information when answering user questions that require more current or specific data.

### Streamlit Interface

The Streamlit app provides a conversational interface. Users input their queries, and the agent processes and responds accordingly. The app maintains session state, keeping track of the conversation history.

![LangGraph Structure](images/UI.png)

## Getting Started

### Prerequisites

- Python 3.12 or higher
- Generate [Tavily API key](https://app.tavily.com/)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Add your `TAVILY_API_KEY` at the top of `agent.py` file to access web search capabilities

## Usage
- Run the Streamlit app:
   ```bash
   streamlit run streamlit-app.py
   ```

- Open the web interface at the address provided by Streamlit (typically `http://localhost:8501`).

- Start chatting with the AI assistant by typing your questions or concerns.

The assistant will either answer your question directly or, if necessary, retrieve additional information from the web to provide a more comprehensive response.