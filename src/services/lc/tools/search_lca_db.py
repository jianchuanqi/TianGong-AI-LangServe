import os
import re
from typing import Optional, Type
from src.config.config import XATA_API_KEY, XATA_LCA_DB_URL
from dotenv import load_dotenv
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import BaseTool
from pydantic import BaseModel
from xata.client import XataClient

load_dotenv()

xata = XataClient(api_key=XATA_API_KEY, db_url=XATA_LCA_DB_URL)

xata_branch="main"


class SearchLCADB(BaseTool):
    name = "search_lca_tool"
    description = "Use original query to search in the Life Cycle Assessment Database."

    class InputSchema(BaseModel):
        query: str

    args_schema: Type[BaseModel] = InputSchema

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool synchronously."""
        results = xata.data().search_branch(
            branch_name=xata_branch, payload={"query": query}
        )

        docs = results["records"]

        return docs

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        results = xata.data().search_branch(
            branch_name=xata_branch, payload={"query": query}
        )

        docs = results["records"]

        return docs


class SearchXataFilterClassification(BaseTool):
    name = "search_database_with_classification_as_filter"
    description = (
        "Search in the Life Cycle Assessment Database if the target table is specified."
    )

    class InputSchema(BaseModel):
        cas: str
        category: str

    args_schema: Type[BaseModel] = InputSchema

    def _run(
        self,
        cas: str,
        category: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool synchronously."""
        results = xata.data().search_branch(
            branch_name=xata_branch,
            payload={
                "query": category,
                "tables": [
                    {
                        "table": "flow",
                        "filter": {"cas_number": cas},
                        "target": ["elementary_flow_categorization"],
                    }
                ],
            },
        )

        docs = results

        return docs

    async def _arun(
        self,
        cas: str,
        category: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        results = xata.data().search_branch(
            branch_name=xata_branch,
            payload={
                "query": category,
                "tables": [
                    {
                        "table": "flow",
                        "filter": {"cas_number": cas},
                        "target": ["elementary_flow_categorization"],
                    }
                ],
            },
        )

        docs = results

        return docs


class SearchXataAsk(BaseTool):
    name = "search_database_by_ask"
    description = (
        "Search in the Life Cycle Assessment Database if the target table is specified."
    )

    class InputSchema(BaseModel):
        query: str

    args_schema: Type[BaseModel] = InputSchema

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool synchronously."""
        results = xata.data().ask(
            "flow",
            query,
            options={
                "searchType": "vector",
                "vectorSearch": {
                    "contentColumn": {"base_name", "elementary_flow_categorization"},
                },
            },
        )

        docs = results

        return docs

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        results = xata.data().ask(
            "flow",
            query,
            options={
                "searchType": "vector",
                "vectorSearch": {
                    "contentColumn": {"base_name", "elementary_flow_categorization"},
                },
            },
        )

        docs = results

        return docs
