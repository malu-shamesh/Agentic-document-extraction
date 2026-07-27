"""Build a tool-calling agent that extracts information from documents using an OCR tool.

Shared between Lab 1 (Tesseract) and Lab 2 (PaddleOCR): the LLM decides when to
call the OCR tool, reasons over its raw output, and answers the user's extraction
task - correcting some OCR noise but still bounded by what the OCR tool actually read.
"""

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

DEFAULT_MODEL = "gpt-5-mini"


def build_agent(tool, tool_description: str, model: str = DEFAULT_MODEL) -> AgentExecutor:
    llm = ChatOpenAI(model=model, temperature=1)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant designed to extract information from documents. "
            f"You have access to this tool: {tool_description}",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, [tool], prompt)
    return AgentExecutor(agent=agent, tools=[tool], verbose=True)


def run_extraction(agent_executor: AgentExecutor, task: str) -> str:
    response = agent_executor.invoke({"input": task})
    return response["output"]
