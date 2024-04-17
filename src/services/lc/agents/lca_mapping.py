import os

from dotenv import load_dotenv
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_openai import ChatOpenAI
from langchain.memory import XataChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_tool
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.services.lc.tools.search_internet_tool import SearchInternet
from src.services.lc.tools.search_lca_db import (
    SearchLCADB,
    SearchXataFilterClassification,
    SearchXataAsk,
    QueryTableFlow,
)
from src.services.lc.tools.search_cas_api_tool import SearchCASApi
from src.config.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    XATA_API_KEY,
    XATA_MEMORY_DB_URL,
    XATA_MEMORY_TABLE_NAME,
)


def init_chat_history(session_id: str) -> BaseChatMessageHistory:
    return XataChatMessageHistory(
        session_id=session_id,
        api_key=XATA_API_KEY,
        db_url=XATA_MEMORY_DB_URL,
        table_name=XATA_MEMORY_TABLE_NAME,
    )


def flow_mapping_internet():
    # lc_tools = [SearchInternet(), SearchVectorDB(), SearchLCADB(), SearchESG()]
    lc_tools = [SearchInternet()]
    oai_tools = [format_tool_to_openai_tool(tool) for tool in lc_tools]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant designed to output JSON."),
            (
                "human",
                "{input}",
            ),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        temperature=0,
        model=OPENAI_MODEL,
    )

    agent = (
        {
            "input": lambda x: x["input"].encode("utf-8").decode("unicode_escape"),
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm.bind(tools=oai_tools)
        | OpenAIToolsAgentOutputParser()
    )

    agent_executor = AgentExecutor(
        agent=agent, tools=lc_tools, verbose=True, handle_parsing_errors=True
    )

    return agent_executor


def flow_mapping_cas_retrieving():
    # lc_tools = [SearchInternet(), SearchVectorDB(), SearchLCADB(), SearchESG()]
    lc_tools = [SearchCASApi()]
    oai_tools = [format_tool_to_openai_tool(tool) for tool in lc_tools]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant. Use HttpRequestGet tool to get the CAS number from CAS API. Respond ONLY the most common ONE CAS number, and add "0"s at the beginning to make the cas number a 11-digit string. Respond "None" if no CAS number is found.""",
            ),
            (
                "human",
                "{input}",
            ),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        temperature=0,
        model=OPENAI_MODEL,
    )

    agent = (
        {
            "input": lambda x: x["input"].encode("utf-8").decode("unicode_escape"),
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm.bind(tools=oai_tools)
        | OpenAIToolsAgentOutputParser()
    )

    agent_executor = AgentExecutor(
        agent=agent, tools=lc_tools, verbose=True, handle_parsing_errors=True
    )

    return agent_executor


def flow_mapping_cas():
    # lc_tools = [SearchInternet(), SearchVectorDB(), SearchLCADB(), SearchESG()]
    lc_tools = [SearchXataFilterClassification()]
    oai_tools = [format_tool_to_openai_tool(tool) for tool in lc_tools]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant designed to output JSON."),
            (
                "human",
                """Search in the Life Cycle Assessment Database and return "base_name", "elementary_flow_categorization", "cas_number", and "uuid" of top 3 records with the highest score. The CAS number is: {cas}. The query category is <{category}>.""",
            ),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        temperature=0,
        model=OPENAI_MODEL,
    )

    agent = (
        {
            "cas": lambda x: x["cas"].encode("utf-8").decode("unicode_escape"),
            "category": lambda x: x["category"]
            .encode("utf-8")
            .decode("unicode_escape"),
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm.bind(tools=oai_tools)
        | OpenAIToolsAgentOutputParser()
    )

    agent_executor = AgentExecutor(
        agent=agent, tools=lc_tools, verbose=True, handle_parsing_errors=True
    )

    return agent_executor


def flow_mapping_synonyms():
    # lc_tools = [SearchInternet(), SearchVectorDB(), SearchLCADB(), SearchESG()]
    lc_tools = [SearchXataAsk()]
    oai_tools = [format_tool_to_openai_tool(tool) for tool in lc_tools]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant designed to output JSON. Use the SearchXataAsk tool to retrieve data.""",
            ),
            (
                "human",
                # "{input}",
                """List the records of carbon dioxide from coal combustion, and the second level of elementary flow categorization falls on "Emissions to air, unspecified". """,
            ),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        temperature=0,
        model=OPENAI_MODEL,
    )

    agent = (
        {
            "input": lambda x: x["input"].encode("utf-8").decode("unicode_escape"),
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm.bind(tools=oai_tools)
        | OpenAIToolsAgentOutputParser()
    )

    agent_executor = AgentExecutor(
        agent=agent, tools=lc_tools, verbose=True, handle_parsing_errors=True
    )

    return agent_executor


def flow_query():
    lc_tools = [QueryTableFlow(), SearchLCADB()]
    oai_tools = [format_tool_to_openai_tool(tool) for tool in lc_tools]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant designed to output JSON. Use the QueryTableFlow tool to retrieve data if a CAS number is provided.Use the SearchLCADB tool to retrieve data if no CAS number is provided. You MUST keep the original CAS number and category string invoking tools. """,
            ),
            (
                "human",
                """Search in the Life Cycle Assessment Database and return "base_name", "elementary_flow_categorization", "cas_number", and "uuid" of top 5 records with the highest score. The CAS number is {cas}. The category is <{category}>. The synonyms are {synonyms}. Do not give any extra information, just the requested fields in the JSON format.""",
            ),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        temperature=0,
        model=OPENAI_MODEL,
    )

    agent = (
        {
            "cas": lambda x: x["cas"].encode("utf-8").decode("unicode_escape"),
            "synonyms": lambda x: x["synonyms"].encode("utf-8").decode("unicode_escape"),
            "category": lambda x: x["category"]
            .encode("utf-8")
            .decode("unicode_escape"),
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm.bind(tools=oai_tools)
        | OpenAIToolsAgentOutputParser()
    )

    agent_executor = AgentExecutor(
        agent=agent, tools=lc_tools, verbose=True, handle_parsing_errors=True
    )

    return agent_executor
