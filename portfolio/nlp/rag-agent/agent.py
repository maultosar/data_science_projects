import os
import pathlib
import ast
import chromadb

from typing import TypedDict, Annotated, List

from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage
from langchain_core.tools import tool

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tavily import TavilyClient

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.core.postprocessor import LLMRerank

os.environ["TAVILY_API_KEY"] = ''
os.chdir(os.path.dirname(__file__))

class AgentState(TypedDict):
    request: str

    action: str
    last_node: str

    rag_queries: List[str]
    web_queries: List[str]

    rag_search_results: str
    web_search_results: str

    knowledge_search_summary: str
    knowledge_search_failure_point: str
    knowledge_reevaluation_counter: int

    messages: Annotated[list[AnyMessage], add_messages]

class Queries(BaseModel):
    """Queries for RAG and web search based on user request"""  
    rag_queries: str = Field(description="A list of queries for RAG search")
    web_queries: str = Field(description="A list of queries for web search")

class RagQueries(BaseModel):
    """Queries for RAG search based on user request""" 
    rag_queries: str = Field(description="A list of queries for RAG search")

class WebQueries(BaseModel):
    """Queries for web search based on user request"""
    web_queries: str = Field(description="A list of queries for web search")

class PsyAgent:
    PSYCHOLOGY_AGENT_PROMPT = """You are a highly qualified and experienced psychologist, psychotherapist, and psychiatrist.\
        Your role is to combine deep theoretical knowledge with practical therapeutic skills, adhering to professional, ethical, and clinical guidelines in every interaction."""

    THERAPIST_POLICY_PROMPT = pathlib.Path('therapist-policy.txt').read_text()

    ACTION_DETECTION_PROMPT = """<INSTRUCTIONS_START>\
        Understand clearly user request and if you are not quite sure what exactly user wants, return \'clarify\'. Make clear understanding a priority!\
        If everything is clear and you think some additional knowledge is needed to give best answer, return \'knowledge_retrieval\'.\
        If everything is clear and you already have all knowledge needed to address request, return \'question_answering\'.\
        If user provided response which implies all his request are resolved, return \'end\'.\
        Remember, respond with one word only as in description above!\
        <INSTRUCTIONS_END>\
        
        <KNOWLEDGE_START>\
        Your current knowledge in memory:\ 
        {knowledge}\
        <KNOWLEDGE_END>"""

    CLARIFICATION_PROMPT = """<INSTRUCTIONS_START>\
        The user\'s intent in his request could be interpreted in many ways.\
        Ask user to specify exactly what he wants potentially giving him options.\
        <INSTRUCTIONS_END>\
        
        <KNOWLEDGE_START>\
        User request: {prompt}\
        <KNOWLEDGE_END>"""
    
    QUERIES_GENERATION_PROMPT = """<INSTRUCTIONS_START>Generate maximum 3 queries for RAG search and also maximum 3 queries for web search based on user request. Formulate queries in a way that they are most likely to return relevant information.<INSTRUCTIONS_END>"""
    
    RAG_RENENERATION_PROMPT = """<INSTRUCTIONS_START>\
        Following queries were generated for RAG based on user request, yet provided results which were not that relevant.\
        Regenerate queries for RAG based on user request. Formulate queries in a way that they are most likely to return relevant information.\
        <INSTRUCTIONS_END>\
        
        <KNOWLEDGE_START>\
        RAG queries: {rag}\
        User request: {request}\
        <KNOWLEDGE_END>"""
    
    WEB_REGENERATION_PROMPT = """<INSTRUCTIONS_START>\
        Following queries were generated for web search based on user request, yet provided results which were not that relevant.\
        Regenerate queries for web search based on user request. Formulate queries in a way that they are most likely to return relevant information.\
        <INSTRUCTIONS_END>\
            
        <KNOWLEDGE_START>\
        Web queries: {web}\
        User request: {request}\ 
        <KNOWLEDGE_END>"""
    
    QUERIES_REGENERATION_PROMPT = """<INSTRUCTIONS_START>\
        Following queries were generated for RAG and web search based on user request, yet provided results which were not that relevant.\
        Regenerate queries for RAG and web search based on user request. Formulate queries in a way that they are most likely to return relevant information.\
        <INSTRUCTIONS_END>\
        
        <KNOWLEDGE_START>\
        RAG queries: {rag}\
        Web queries: {web}\
        User request: {request}\
        <KNOWLEDGE_END>"""

    KNOWLEDGE_RELEVANCY_EVALUATION_PROMPT = """<INSTRUCTIONS_START>\
        Evaluate if information retrieved from RAG and web search is relevant enough to user request.\
        If both RAG and web are relevant, return \'none\'.\
        If both RAG and web are irrelevant, return \'both\'.\
        If RAG is irrelevant and web is relevant, return \'rag\'.\
        If RAG is relevant and web is irrelevant, return \'web\'.\
        Remember, respond with only one word!\
        <INSTRUCTIONS_END>\
        
        <KNOWLEDGE_START>\
        User request: {request}\
        RAG search results: {rag}\
        Web search results: {web}\
        <KNOWLEDGE_END>"""
    
    KNOWLEDGE_SUMMARY_PROMPT = """<INSTRUCTIONS_START>\
        Filter, order and summarize information retrieved from RAG and web search, so only information relevant to user request in conversation history context is left.\
        <INSTRUCTIONS_END>\
        
        <KNOWLEDGE_START>\
        User request: {request}\
        RAG search results: {rag}\
        Web search results: {web}\
        <KNOWLEDGE_END>"""

    QUESTION_ANSWERING_PROMPT = """<INSTRUCTIONS_START>\
        Answer user request according to policy.\
        Take into account the conversation history as well.\
        Talk to user in a natural manner, he doesn't need to know about your internal workings.\
        <INSTRUCTIONS_END>\
        
        <KNOWLEDGE_START>\
        Here is additional information from sources to help with request: {knowledge_summary}\
        User request: {request}\
        <KNOWLEDGE_END>"""
        

    # Used OpenAI interface as it offers more features than ChatOllama
    default_model = ChatOpenAI(
        api_key="ollama",
        model="llama3.1",
        base_url="http://localhost:11434/v1",
        temperature=0
    )
    memory = SqliteSaver.from_conn_string(":memory:")
    tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])



    def __init__(self, config, model=default_model, debug=False):
        self.model = model
        
        builder = StateGraph(AgentState)

        builder.add_node("action_selector", self.action_selector_node)
        builder.add_node("clarify", self.clarify_node)
        builder.add_node("knowledge_retrieval", self.knowledge_retrieval_node)
        builder.add_node("rag", self.rag_search_node)
        builder.add_node("web", self.web_search_node)
        builder.add_node("knowledge_evaluation", self.knowledge_evaluation_node)
        builder.add_node("knowledge_summary", self.knowledge_summary_node)
        builder.add_node("question_answering", self.question_answering_node)

        builder.set_entry_point("action_selector")

        builder.add_conditional_edges(
            "action_selector",
            self.select_action,
            {"clarify": "clarify", "knowledge_retrieval": "knowledge_retrieval", "question_answering": "question_answering", "end": END}
        )

        builder.add_edge("clarify", "action_selector")
        builder.add_edge("question_answering", "action_selector")
        builder.add_edge("knowledge_retrieval", "rag")
        builder.add_edge("knowledge_retrieval", "web")
        builder.add_edge("rag", "knowledge_evaluation")
        builder.add_edge("web", "knowledge_evaluation")
        builder.add_edge("knowledge_summary", "question_answering")

        builder.add_conditional_edges(
            "knowledge_evaluation",
            self.knowledge_relevancy_evaluation,
            {"question_answering": "question_answering", "knowledge_retrieval": "knowledge_retrieval", "knowledge_summary": "knowledge_summary"}
        )

        self.graph = builder.compile(checkpointer=self.memory, interrupt_after=["clarify", "question_answering"], debug=debug)
        self.graph.update_state(config, {"knowledge_reevaluation_counter": 0})

        self.__initialize_vector_store()

    def __initialize_vector_store(self):
        Settings.llm = Ollama(model="llama3.1")
        documents = SimpleDirectoryReader("./knowledge_base", filename_as_id=True).load_data()
        
        db = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db.get_or_create_collection("knowledge_base")

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        embedding_model = OllamaEmbedding(
            model_name="mxbai-embed-large",
            base_url="http://localhost:11434")

        if not os.path.exists("./chroma_db"):
            vector_store_index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embedding_model)
            self.vector_index = vector_store_index.as_query_engine()
        else:
            vector_store_index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embedding_model)
            vector_store_index.refresh_ref_docs(documents)
            self.vector_index = vector_store_index.as_query_engine()

    def action_selector_node(self, state: AgentState):
        messages = state["messages"] + [
            SystemMessage(content=self.PSYCHOLOGY_AGENT_PROMPT),
            SystemMessage(content=self.ACTION_DETECTION_PROMPT.format(knowledge=state["knowledge_search_summary"])),
            state["messages"][-1]
        ] 
        response = self.model.invoke(messages)
        
        return {"request": state["messages"][-1].content, 
                "action": response.content, 
                "last_node": "action_selector"}

    def select_action(self, state: AgentState):
        return state["action"]

    def clarify_node(self, state: AgentState):
        messages = [
            SystemMessage(content=self.PSYCHOLOGY_AGENT_PROMPT),
            HumanMessage(content=self.CLARIFICATION_PROMPT.format(prompt=state['request'])), 
        ]
        
        response = self.model.invoke(messages)
        return {"messages": [response], "last_node": "clarify"}

    def knowledge_retrieval_node(self, state: AgentState):
        rag_queries = state["rag_queries"]
        web_queries = state["web_queries"]

        if state["knowledge_search_failure_point"] == "both":
            queries = self.model.with_structured_output(Queries).invoke([
                SystemMessage(content=self.QUERIES_REGENERATION_PROMPT.format(rag=state["rag_queries"], web=state["web_queries"], request=state["request"])),
                HumanMessage(content=state["request"])
            ])
            rag_queries = ast.literal_eval(queries.rag_queries) if queries.rag_queries else []
            web_queries = ast.literal_eval(queries.web_queries) if queries.web_queries else []
        elif state["knowledge_search_failure_point"] == "rag":
            queries = self.model.with_structured_output(RagQueries).invoke([
                SystemMessage(content=self.RAG_RENENERATION_PROMPT.format(rag=state["rag_queries"], request=state["request"])),
                HumanMessage(content=state["request"])
            ])
            rag_queries = ast.literal_eval(queries.rag_queries) if queries.rag_queries else []
        elif state["knowledge_search_failure_point"] == "web":
            queries = self.model.with_structured_output(WebQueries).invoke([
                SystemMessage(content=self.WEB_REGENERATION_PROMPT.format(web=state["web_queries"], request=state["request"])),
                HumanMessage(content=state["request"])
            ])
            web_queries = ast.literal_eval(queries.web_queries) if queries.web_queries else []
        else:
            queries = self.model.with_structured_output(Queries).invoke([
                SystemMessage(content=self.QUERIES_GENERATION_PROMPT),
                HumanMessage(content=state["request"])
            ])
            rag_queries = ast.literal_eval(queries.rag_queries) if queries.rag_queries else []
            web_queries = ast.literal_eval(queries.web_queries) if queries.web_queries else []
        return {"rag_queries": rag_queries, "web_queries": web_queries, "last_node": "knowledge_retrieval"}

    def rag_search_node(self, state: AgentState):
        """RAG search through local documents vector database based on user request"""

        if not state["knowledge_search_failure_point"] or not state["knowledge_search_failure_point"] == "web":
            rag_results = []
            for rag_search_query in state["rag_queries"]:
                result = self.vector_index.query(rag_search_query)
                rag_results.append(result.response)
            return {"rag_search_results": rag_results}

    def web_search_node(self, state: AgentState):
        """Searches web for information based on user's request"""
        
        if not state["knowledge_search_failure_point"] or not state["knowledge_search_failure_point"] == "rag":
            search_results = []
            for q in state["web_queries"]:
                response = self.tavily.search(query=q, max_results=2)
                for r in response['results']:
                    search_results.append(r['content'])
            return {"web_search_results": search_results}

    def knowledge_evaluation_node(self, state: AgentState):
        messages = state["messages"] + [
            SystemMessage(content=self.PSYCHOLOGY_AGENT_PROMPT),
            HumanMessage(content=self.KNOWLEDGE_RELEVANCY_EVALUATION_PROMPT.format(rag=state["rag_search_results"], web=state["web_search_results"], request=state["request"])) 
        ]

        response = self.model.invoke(messages)

        failure_point = response.content
        counter = state["knowledge_reevaluation_counter"]
        counter += 1
        web_search_results = state["web_search_results"]
        rag_search_results = state["rag_search_results"]

        if failure_point == "web":
            web_search_results = []
        elif failure_point == "rag":
            rag_search_results = []
        elif failure_point == "both":
            web_search_results = []
            rag_search_results = []
        
        return {"knowledge_search_failure_point": failure_point,
                "knowledge_reevaluation_counter": counter,
                "web_search_results": web_search_results,
                "rag_search_results": rag_search_results,
                "last_node": "knowledge_evaluation"}

    def knowledge_relevancy_evaluation(self, state: AgentState):
        failure_point = state["knowledge_search_failure_point"]
        counter = state["knowledge_reevaluation_counter"]

        if not failure_point == "none" and counter <= 3:
            return "knowledge_retrieval"
        elif failure_point == "both":
            return "question_answering"
        else:
            return "knowledge_summary"
        
    def knowledge_summary_node(self, state: AgentState):
        rag_search_results = '\n'.join(state["rag_search_results"])
        web_search_results = '\n'.join(state["web_search_results"])

        messages = state["messages"] + [
            SystemMessage(content=self.PSYCHOLOGY_AGENT_PROMPT),
            HumanMessage(content=self.KNOWLEDGE_SUMMARY_PROMPT.format(rag=rag_search_results, web=web_search_results, request=state["request"])) 
        ]

        response = self.model.invoke(messages)

        return {"knowledge_search_summary": response.content, 
                "knowledge_reevaluation_counter": 0, 
                "knowledge_search_failure_point": None,
                "rag_search_results": [],
                "web_search_results": [], 
                "last_node": "knowledge_summary"}

    def question_answering_node(self, state: AgentState):
        messages = state["messages"] + [
            SystemMessage(content=self.PSYCHOLOGY_AGENT_PROMPT),
            SystemMessage(content=self.THERAPIST_POLICY_PROMPT),
            HumanMessage(content=self.QUESTION_ANSWERING_PROMPT.format(knowledge_summary=state["knowledge_search_summary"], request=state["request"])), 
        ]
        
        response = self.model.invoke(messages)
        return {"messages": [response], "" "last_node": "question_answering"}

    def draw_graph(self):
        from IPython.display import Image
        return Image(self.graph.get_graph().draw_mermaid_png(output_file_path="./graph.png"))
    
    def initial_invocation(self, input, config):
        user_input = HumanMessage(content=input)
        for event in self.graph.stream({"messages": [user_input]}, config, stream_mode="values"):
            if event["messages"]:
                event["messages"][-1].pretty_print()
    
    def human_assisted_input_loop(self, input, config):
        while True:
            user_response = HumanMessage(content=input)

            last_node = self.graph.get_state(config).values["last_node"]
            self.graph.update_state(config, {"messages": [user_response]}, as_node=last_node)
            
            for event in self.graph.stream(None, config, stream_mode="values"):
                if event["messages"]:
                    event["messages"][-1].pretty_print()