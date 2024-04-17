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
import json

load_dotenv()

xata = XataClient(api_key=XATA_API_KEY, db_url=XATA_LCA_DB_URL)

xata_branch = "main"


class SearchLCADB(BaseTool):
    name = "search_lca_db"
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


class QueryTableFlow(BaseTool):
    name = "query_table_flow_in_lca_db"
    description = "Use original query to search in the Life Cycle Assessment Database."

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
        category_full = find_category_hierarchy(category)

        results = xata.data().query(
            table_name="flow",
            payload={
                "filter": {
                    "cas_number": cas,
                    "elementary_flow_categorization": category_full,
                }
            },
        )

        docs = results["records"]

        return docs

    async def _arun(
        self,
        cas: str,
        category: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool synchronously."""
        category_full = find_category_hierarchy(category)

        results = xata.data().query(
            table_name="flow",
            payload={
                "filter": {
                    "cas_number": cas,
                    "elementary_flow_categorization": category_full,
                }
            },
        )

        docs = results["records"]

        return docs


# Predefined category relationships
category_relationships = [
    {
        "category_name": [
            {"category_name": "Emissions", "level": 0},
            {"category_name": "Emissions to water", "level": 1},
            {"category_name": "Emissions to fresh water", "level": 2},
        ]
    },
    {
        "category_name": [
            {"category_name": "Emissions", "level": 0},
            {"category_name": "Emissions to water", "level": 1},
            {"category_name": "Emissions to sea water", "level": 2},
        ]
    },
    {
        "category_name": [
            {"category_name": "Emissions", "level": 0},
            {"category_name": "Emissions to water", "level": 1},
            {"category_name": "Emissions to water, unspecified", "level": 2},
        ]
    },
    {
        "category_name": [
            {"category_name": "Emissions", "level": 0},
            {"category_name": "Emissions to water", "level": 1},
            {
                "category_name": "Emissions to water, unspecified (long-term)",
                "level": 2,
            },
        ]
    },
    {
        "category_name": [
            {"category_name": "Emissions", "level": 0},
            {"category_name": "Emissions to soil", "level": 1},
            {"category_name": "Emissions to agricultural soil", "level": 2},
        ]
    },
    {
        "category_name": [
            {"category_name": "Emissions", "level": 0},
            {"category_name": "Emissions to soil", "level": 1},
            {"category_name": "Emissions to non-agricultural soil", "level": 2},
        ]
    },
    {
        "category_name": [
            {"category_name": "Emissions", "level": 0},
            {"category_name": "Emissions to soil", "level": 1},
            {"category_name": "Emissions to soil, unspecified", "level": 2},
        ]
    },
    {
        "category_name": [
            {"category_name": "Emissions", "level": 0},
            {"category_name": "Emissions to soil", "level": 1},
            {"category_name": "Emissions to soil, unspecified (long-term)", "level": 2},
        ]
    },
    {
        "category_name": [
            {"category_name": "Emissions", "level": 0},
            {"category_name": "Emissions to air", "level": 1},
            {"category_name": "Emissions to urban air close to ground", "level": 2},
        ]
    },
    {
        "category_name": [
            {"category_name": "Emissions", "level": 0},
            {"category_name": "Emissions to air", "level": 1},
            {
                "category_name": "Emissions to non-urban air or from high stacks",
                "level": 2,
            },
        ]
    },
    {
        "category_name": [
            {"category_name": "Emissions", "level": 0},
            {"category_name": "Emissions to air", "level": 1},
            {
                "category_name": "Emissions to lower stratosphere and upper troposphere",
                "level": 2,
            },
        ]
    },
    {
        "category_name": [
            {"category_name": "Emissions", "level": 0},
            {"category_name": "Emissions to air", "level": 1},
            {"category_name": "Emissions to air, unspecified", "level": 2},
        ]
    },
    {
        "category_name": [
            {"category_name": "Emissions", "level": 0},
            {"category_name": "Emissions to air", "level": 1},
            {"category_name": "Emissions to air, unspecified (long-term)", "level": 2},
        ]
    },
    {
        "category_name": [
            {"category_name": "Resources", "level": 0},
            {"category_name": "Resources from ground", "level": 1},
            {
                "category_name": "Non-renewable material resources from ground",
                "level": 2,
            },
        ]
    },
    {
        "category_name": [
            {"category_name": "Resources", "level": 0},
            {"category_name": "Resources from ground", "level": 1},
            {
                "category_name": "Non-renewable element resources from ground",
                "level": 2,
            },
        ]
    },
    {
        "category_name": [
            {"category_name": "Resources", "level": 0},
            {"category_name": "Resources from ground", "level": 1},
            {"category_name": "Non-renewable energy resources from ground", "level": 2},
        ]
    },
    {
        "category_name": [
            {"category_name": "Resources", "level": 0},
            {"category_name": "Resources from ground", "level": 1},
            {"category_name": "Renewable element resources from ground", "level": 2},
        ]
    },
    {
        "category_name": [
            {"category_name": "Resources", "level": 0},
            {"category_name": "Resources from ground", "level": 1},
            {"category_name": "Renewable energy resources from ground", "level": 2},
        ]
    },
    {
        "category_name": [
            {"category_name": "Resources", "level": 0},
            {"category_name": "Resources from ground", "level": 1},
            {"category_name": "Renewable material resources from ground", "level": 2},
        ]
    },
    {
        "category_name": [
            {"category_name": "Resources", "level": 0},
            {"category_name": "Resources from ground", "level": 1},
            {
                "category_name": "Renewable resources from ground, unspecified",
                "level": 2,
            },
        ]
    },
    {
        "category_name": [
            {"category_name": "Resources", "level": 0},
            {"category_name": "Resources from ground", "level": 1},
            {
                "category_name": "Non-renewable resources from ground, unspecified",
                "level": 2,
            },
        ]
    },
    {
        "category_name": [
            {"category_name": "Resources", "level": 0},
            {"category_name": "Resources from water", "level": 1},
            {
                "category_name": "Non-renewable material resources from water",
                "level": 2,
            },
        ]
    },
    {
        "category_name": [
            {"category_name": "Resources", "level": 0},
            {"category_name": "Resources from water", "level": 1},
            {"category_name": "Non-renewable element resources from water", "level": 2},
        ]
    },
    {
        "category_name": [
            {"category_name": "Resources", "level": 0},
            {"category_name": "Resources from water", "level": 1},
            {"category_name": "Non-renewable energy resources from water", "level": 2},
        ]
    },
    {
        "category_name": [
            {"category_name": "Resources", "level": 0},
            {"category_name": "Resources from water", "level": 1},
            {"category_name": "Renewable element resources from water", "level": 2},
        ]
    },
    {
        "category_name": [
            {"category_name": "Resources", "level": 0},
            {"category_name": "Resources from water", "level": 1},
            {"category_name": "Renewable energy resources from water", "level": 2},
        ]
    },
    {
        "category_name": [
            {"category_name": "Resources", "level": 0},
            {"category_name": "Resources from water", "level": 1},
            {"category_name": "Renewable material resources from water", "level": 2},
        ]
    },
    {
        "category_name": [
            {"category_name": "Resources", "level": 0},
            {"category_name": "Resources from water", "level": 1},
            {
                "category_name": "Renewable resources from water, unspecified",
                "level": 2,
            },
        ]
    },
    {
        "category_name": [
            {"category_name": "Resources", "level": 0},
            {"category_name": "Resources from water", "level": 1},
            {
                "category_name": "Non-renewable resources from water, unspecified",
                "level": 2,
            },
        ]
    },
    {
        "category_name": [
            {"category_name": "Resources", "level": 0},
            {"category_name": "Resources from air", "level": 1},
            {"category_name": "Non-renewable material resources from air", "level": 2},
        ]
    },
    {
        "category_name": [
            {"category_name": "Resources", "level": 0},
            {"category_name": "Resources from air", "level": 1},
            {"category_name": "Non-renewable element resources from air", "level": 2},
        ]
    },
    {
        "category_name": [
            {"category_name": "Resources", "level": 0},
            {"category_name": "Resources from air", "level": 1},
            {"category_name": "Non-renewable energy resources from air", "level": 2},
        ]
    },
    {
        "category_name": [
            {"category_name": "Resources", "level": 0},
            {"category_name": "Resources from air", "level": 1},
            {"category_name": "Renewable element resources from air", "level": 2},
        ]
    },
    {
        "category_name": [
            {"category_name": "Resources", "level": 0},
            {"category_name": "Resources from air", "level": 1},
            {"category_name": "Renewable energy resources from air", "level": 2},
        ]
    },
    {
        "category_name": [
            {"category_name": "Resources", "level": 0},
            {"category_name": "Resources from air", "level": 1},
            {"category_name": "Renewable material resources from air", "level": 2},
        ]
    },
    {
        "category_name": [
            {"category_name": "Resources", "level": 0},
            {"category_name": "Resources from air", "level": 1},
            {"category_name": "Renewable resources from air, unspecified", "level": 2},
        ]
    },
    {
        "category_name": [
            {"category_name": "Resources", "level": 0},
            {"category_name": "Resources from air", "level": 1},
            {
                "category_name": "Non-renewable resources from air, unspecified",
                "level": 2,
            },
        ]
    },
    {
        "category_name": [
            {"category_name": "Resources", "level": 0},
            {"category_name": "Resources from biosphere", "level": 1},
            {"category_name": "Renewable element resources from biosphere", "level": 2},
        ]
    },
    {
        "category_name": [
            {"category_name": "Resources", "level": 0},
            {"category_name": "Resources from biosphere", "level": 1},
            {"category_name": "Renewable energy resources from biosphere", "level": 2},
        ]
    },
    {
        "category_name": [
            {"category_name": "Resources", "level": 0},
            {"category_name": "Resources from biosphere", "level": 1},
            {
                "category_name": "Renewable material resources from biosphere",
                "level": 2,
            },
        ]
    },
    {
        "category_name": [
            {"category_name": "Resources", "level": 0},
            {"category_name": "Resources from biosphere", "level": 1},
            {"category_name": "Renewable genetic resources from biosphere", "level": 2},
        ]
    },
    {
        "category_name": [
            {"category_name": "Resources", "level": 0},
            {"category_name": "Resources from biosphere", "level": 1},
            {
                "category_name": "Renewable resources from biosphere, unspecified",
                "level": 2,
            },
        ]
    },
]


def find_category_hierarchy(input_category):
    # Iterate through each category list to find the match
    for category in category_relationships:
        for cat in category["category_name"]:
            if cat["category_name"] == input_category and cat["level"] == 2:
                # Convert the hierarchy to a JSON string if found
                return json.dumps([category])
    # Return an error message if the category is not found
    return json.dumps({"error": "Category not found"})
