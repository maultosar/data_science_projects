import os
import sys
import pytest
import yaml
from trulens.core.session import TruSession
import numpy as np
from trulens.apps.llamaindex import TruLlama
from trulens.core import Feedback
import litellm
from trulens.providers.litellm import LiteLLM
from trulens.dashboard import run_dashboard
from src.agent import PsyAgent

litellm.set_verbose = False

RESOURCE_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources/rag"))
EVAL_QUESTIONS_FILE = f"{RESOURCE_FOLDER}/test-questions.yml"

@pytest.fixture(scope="module")
def eval_questions():
    with open(EVAL_QUESTIONS_FILE, 'r') as file:
        data = yaml.safe_load(file)
        return data['questions']

@pytest.fixture(scope="module")
def agent():
    config = {"configurable": {"thread_id": "1"}}
    return PsyAgent(config, knowledge_base_folder=f"{RESOURCE_FOLDER}/pdf", web_search_enabled=False, debug=True)

@pytest.fixture(scope="module")
def query_engines(agent):
    return {
        'automerging_engine': agent.automerging_query_engine,
        'rerank_engine': agent.rerank_query_engine,
        'basic_engine': agent.query_engine
    }

def test_query_engine(agent, query_engines, eval_questions):
    session = TruSession()
    session.reset_database()

    for name, query_engine in query_engines.items():
        provider = LiteLLM(
            model_engine=f"ollama/{agent.TEXT_GENERATION_MODEL_NAME}", api_base="http://localhost:11434/v1"
        )
        context = TruLlama.select_context(query_engine)

        f_groundedness = Feedback(
            provider.groundedness_measure_with_cot_reasons, name="Groundedness"
        ).on(context.collect()).on_output()

        f_answer_relevance = Feedback(
            provider.relevance_with_cot_reasons, name="Answer Relevance"
        ).on_input_output()

        f_context_relevance = Feedback(
            provider.context_relevance_with_cot_reasons, name="Context Relevance"
        ).on_input().on(context).aggregate(np.mean)

        tru_query_engine_recorder = TruLlama(
            query_engine,
            app_name=name,
            app_version="base",
            feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance],
        )

        with tru_query_engine_recorder as recording:
            for question in eval_questions:
                query_engine.query(question)
        
        leaderboard = session.get_leaderboard(app_ids=[tru_query_engine_recorder.app_id])

        if 'Groundedness' in leaderboard.columns:
            assert leaderboard.iloc[0]['Groundedness'] >= 0.8, "Groundedness score is too low"
        if 'Answer Relevance' in leaderboard.columns:
            assert leaderboard.iloc[0]['Answer Relevance'] >= 0.8, "Relevance score is too low"
        if 'Context Relevance' in leaderboard.columns:
            assert leaderboard.iloc[0]['Context Relevance'] >= 0.5, "Context Relevance score is too low"
