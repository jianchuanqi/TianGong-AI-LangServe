from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.memory import XataChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_tool
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI

from src.config.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
)

from src.services.lc.tools.search_internet_tool import SearchInternet
from src.services.lc.tools.search_cas_api_tool import SearchCASApi
from src.services.lc.tools.search_lca_db import (
    SearchLCADB,
    QueryTableFlow,
)


def openai_flow_recommender_runnable():

    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        temperature=0,
        model=OPENAI_MODEL,
    )

    # Step 1: Parse the query to extract the name and category of the substance

    query_parse_tools = [
        {
            "type": "function",
            "function": {
                "name": "Parse_query",
                "description": "You are a world class algorithm for extracting information to output JSON.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The chemical substance name in english without any description.",
                        },
                        "category": {
                            "type": "string",
                            "enum": [
                                "",
                                "emissions to fresh water",
                                "emissions to sea water",
                                "emissions to water, unspecified",
                                "emissions to water, unspecified (long-term)",
                                "emissions to agricultural soil",
                                "emissions to non-agricultural soil",
                                "emissions to soil, unspecified",
                                "emissions to soil, unspecified (long-term)",
                                "emissions to urban air close to ground",
                                "emissions to non-urban air or from high stacks",
                                "emissions to lower stratosphere and upper troposphere",
                                "emissions to air, unspecified",
                                "emissions to air, unspecified (long-term)",
                                "non-renewable material resources from ground",
                                "non-renewable element resources from ground",
                                "non-renewable energy resources from ground",
                                "renewable element resources from ground",
                                "renewable energy resources from ground",
                                "renewable material resources from ground",
                                "renewable resources from ground, unspecified",
                                "non-renewable resources from ground, unspecified",
                                "non-renewable material resources from water",
                                "non-renewable element resources from water",
                                "non-renewable energy resources from water",
                                "renewable element resources from water",
                                "renewable energy resources from water",
                                "renewable material resources from water",
                                "renewable resources from water, unspecified",
                                "non-renewable resources from water, unspecified",
                                "non-renewable material resources from air",
                                "non-renewable element resources from air",
                                "non-renewable energy resources from air",
                                "renewable element resources from air",
                                "renewable energy resources from air",
                                "renewable material resources from air",
                                "renewable resources from air, unspecified",
                                "non-renewable resources from air, unspecified",
                                "renewable element resources from biosphere",
                                "renewable energy resources from biosphere",
                                "renewable material resources from biosphere",
                                "renewable genetic resources from biosphere",
                                "renewable resources from biosphere, unspecified",
                            ],
                            "description": "The category of the substance and leave it blank if not provided.",
                        },
                    },
                    "required": ["name", "category"],
                },
            },
        }
    ]

    query_parse_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a world class algorithm for extracting information to output JSON",
            ),
            (
                "human",
                """Extract information from the bracket below and organize it into JSON with keys of "name" and "category". Respond "None" if no substance name is extracted. <{query}>""",
            ),
            MessagesPlaceholder(variable_name="query_parse_scratchpad"),
        ]
    )

    query_parse = (
        {
            "query": lambda x: x["input"],
            "query_parse_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
        }
        | query_parse_prompt
        | llm.bind_tools(tools=query_parse_tools)
        | OpenAIToolsAgentOutputParser()
    )

    # Step 2: Retrieve the synonyms of the substance name

    synonyms_retrieve_tools = [SearchInternet()]
    synonyms_retrieve_oai_tools = [
        format_tool_to_openai_tool(tool) for tool in synonyms_retrieve_tools
    ]

    synonyms_retrieve_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are helpful AI assistant."),
            (
                "human",
                """List top 5 synonyms of the substance name mentioned the follwing bracket in English and output as JSON. The original name must be listed as one of the synonyms. Respond "None" if no synonyms name is extracted. <{parsed_query}>""",
            ),
            MessagesPlaceholder(variable_name="synonyms_retrieve_scratchpad"),
        ]
    )

    synonyms_retrieve = (
        {
            "parsed_query": query_parse,
            "synonyms_retrieve_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
        }
        | synonyms_retrieve_prompt
        | llm.bind_tools(tools=synonyms_retrieve_oai_tools)
        | OpenAIToolsAgentOutputParser()
    )

    # Step 3: Retrieve the CAS number of the substance

    cas_retrieve_tools = [SearchCASApi()]
    cas_retrieve_oai_tools = [
        convert_to_openai_function(tool) for tool in cas_retrieve_tools
    ]

    cas_retrieve_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant. Use search_cas_api tool to get the CAS number from CAS API. Respond ONLY the most common ONE CAS number, and you will have to add "0"s at the beginning to make the cas number a 11-digit string. Respond "None" if no CAS number is found.""",
            ),
            (
                "human",
                "{retrieved_synonyms}",
            ),
            MessagesPlaceholder(variable_name="cas_retrieve_scratchpad"),
        ]
    )

    cas_retrieve = (
        {
            "retrieved_synonyms": synonyms_retrieve,
            "cas_retrieve_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
        }
        | cas_retrieve_prompt
        | llm.bind_tools(tools=cas_retrieve_oai_tools)
        | OpenAIToolsAgentOutputParser()
    )

    # Step 4: Search flow in the database

    flow_search_tools = [QueryTableFlow(), SearchLCADB()]
    flow_search_oai_tools = [
        format_tool_to_openai_tool(tool) for tool in flow_search_tools
    ]

    flow_search_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant designed to output JSON. Use the query_table_flow_in_lca_db tool to retrieve data if a CAS number is provided. Use the search_lca_db tool to retrieve data if no CAS number is provided. You MUST keep the original CAS number and category string invoking tools. """,
            ),
            (
                "human",
                """Search in the Life Cycle Assessment Database and return "base_name", "elementary_flow_categorization", "cas_number", and "uuid" of top 5 records with the highest score. The CAS number is <{retrieved_cas}>. The category is given in <{parsed_query}>. The synonyms are <{retrieved_synonyms}>. Do not give any extra information, just the requested fields in the JSON format.""",
            ),
            MessagesPlaceholder(variable_name="flow_search_scratchpad"),
        ]
    )

    flow_search = (
        {
            "retrieved_cas": cas_retrieve,
            "retrieved_synonyms": synonyms_retrieve,
            "parsed_query": query_parse,
            "flow_search_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
        }
        | flow_search_prompt
        | llm.bind_tools(tools=flow_search_oai_tools)
        | OpenAIToolsAgentOutputParser()
    )

    # Execute the agent

    agent_executor = AgentExecutor(
        agent=flow_search,
        tools=flow_search_tools,
        verbose=True,
        handle_parsing_errors=True,
    )

    return agent_executor
