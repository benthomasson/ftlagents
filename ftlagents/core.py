import importlib.resources

import yaml
from smolagents import LiteLLMModel

from ftlagents.agents import CodeAgent


def create_model(model_id, context=8192, llm_api_base=None):

    return LiteLLMModel(
        model_id=model_id,
        num_ctx=context,
        api_base=llm_api_base,
    )


def make_agent(tools, model):
    prompt_templates = yaml.safe_load(
        importlib.resources.files("ftlagents.prompts")
        .joinpath("code_agent.yaml")
        .read_text()
    )
    agent = CodeAgent(
        tools=tools,
        model=model,
        verbosity_level=4,
        prompt_templates=prompt_templates,
        additional_authorized_imports=["typing"],
    )
    return agent


def run_agent(tools, model, problem_statement):
    agent = make_agent(tools, model)
    return agent.run(problem_statement, stream=True)
