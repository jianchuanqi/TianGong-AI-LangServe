from typing import Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from langchain.tools import BaseTool
from pydantic import BaseModel
from langchain.utilities import TextRequestsWrapper
import json


class HttpRequestGet(BaseTool):
    name = "http_request_tool"
    description = "Retrieve data via RESTful API with HTTP Request GET."

    class InputSchema(BaseModel):
        query: str

    args_schema: Type[BaseModel] = InputSchema

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool synchronously."""
        # 将输入的字符串转换为URL编码
        api_url = "https://commonchemistry.cas.org/api/search?q="
        url = api_url + query
        requests = TextRequestsWrapper()
        try:
            response = requests.get(url)
            response_json = json.loads(response)
            return response_json["results"][0]["rn"]
        except:
            return "API Error"

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        cas = []
        # 将输入的字符串转换为URL编码
        api_url = "https://commonchemistry.cas.org/api/search?q="
        url = api_url + query
        requests = TextRequestsWrapper()
        try:
            response = requests.get(url)
            response_json = json.loads(response)
            return response_json["results"][0]["rn"]
        except:
            return "API Error"
