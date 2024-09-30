import os
import pytest
import yaml
from unittest.mock import patch
from trulens.core import TruSession, Feedback
from trulens.feedback import GroundTruthAgreement
from trulens.providers.litellm import LiteLLM
from trulens.apps.basic import TruBasicApp
from src.agent import PsyAgent
from langchain_core.messages import HumanMessage
from trulens.dashboard import run_dashboard

RESOURCE_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources/agent"))
YAML_FILE = f"{RESOURCE_FOLDER}/test-question-answer.yml"

config = {"configurable": {"thread_id": "1"}}
agent = PsyAgent(config, knowledge_retrieval=False, debug=True)

@pytest.fixture(scope="module")
def question_answer_pairs():
    with open(YAML_FILE, 'r') as file:
        data = yaml.safe_load(file)
        print(data['questions_answers'])
        return data['questions_answers']

@pytest.fixture(scope="session")
def trulens_session(request):
    session = TruSession()
    session.reset_database()
    return session

def run_agent(question):
    user_prompt = HumanMessage(content=question)
    agent_response = agent.graph.invoke({"messages": [user_prompt]}, config)
    return agent_response['messages'][-1].content

def test_psy_agent_without_knowledge_retrieval(question_answer_pairs, trulens_session):
    provider = LiteLLM(model_engine=f"ollama/{agent.TEXT_GENERATION_MODEL_NAME}", api_base="http://localhost:11434/v1")

    for pair in question_answer_pairs:
        question = pair['question']
        expected_answer = pair['answer']

        golden_set = [{"query": question, "expected_response": expected_answer}]
        f_groundtruth = Feedback(
            GroundTruthAgreement(golden_set, provider=provider).agreement_measure,
            name="Ground Truth Semantic Agreement"
        ).on_input_output()

        f_relevance = Feedback(
                provider.relevance_with_cot_reasons, name="Answer Relevance"
        ).on_input_output()

        tru_agent_recorder = TruBasicApp(
            run_agent, app_name="PsyAgent-Test", feedbacks=[f_groundtruth, f_relevance]
        )

        with tru_agent_recorder as recording:
            tru_agent_recorder.app(question)

    assert trulens_session.get_leaderboard(app_ids=[]).iloc[0]['Ground Truth Semantic Agreement'] >= 0.5
    assert trulens_session.get_leaderboard(app_ids=[]).iloc[0]['Answer Relevance'] >= 0.8
