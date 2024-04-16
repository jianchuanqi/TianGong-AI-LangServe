import os

import toml

config = toml.load(".secrects/secrets.toml")

OPENAI_API_KEY = config["OPENAI"]["API_KEY"]
OPENAI_MODEL = config["OPENAI"]["MODEL"]
OPENAI_EMBEDDING_MODEL_V2 = config["OPENAI"]["EMBEDDING_MODEL_V2"]
OPENAI_EMBEDDING_MODEL_V3 = config["OPENAI"]["EMBEDDING_MODEL_V3"]

ZHIPUAI_API_KEY = config["ZHIPUAI"]["API_KEY"]
ZHIPUAI_MODEL = config["ZHIPUAI"]["MODEL"]

PINECONE_API_KEY = config["PINECONE"]["API_KEY"]
PINECONE_INDEX_NAME = config["PINECONE"]["INDEX_NAME"]
PINECONE_NAMESPACE_SCI = config["PINECONE"]["NAMESPACE_SCI"]
PINECONE_NAMESPACE_PATENT = config["PINECONE"]["NAMESPACE_PATENT"]
PINECONE_NAMESPACE_ESG = config["PINECONE"]["NAMESPACE_ESG"]
PINECONE_NAMESPACE_STANDARD = config["PINECONE"]["NAMESPACE_STANDARD"]

XATA_API_KEY = config["XATA"]["API_KEY"]
XATA_MEMORY_TABLE_NAME = config["XATA"]["MEMORY_TABLE_NAME"]
XATA_MEMORY_DB_URL = config["XATA"]["MEMORY_DB_URL"]
XATA_ESG_DB_URL = config["XATA"]["ESG_DB_URL"]
XATA_DOCS_DB_URL = config["XATA"]["DOCS_DB_URL"]

WEAVIATE_HOST = config["WEAVIATE"]["HOST"]
WEAVIATE_PORT = config["WEAVIATE"]["PORT"]
WEAVIATE_COLLECTION_NAME = config["WEAVIATE"]["COLLECTION_NAME"]

E2B_API_KEY = config["E2B"]["API_KEY"]

WIX_CLIENT_ID = config["WIX"]["WIX_CLIENT_ID"]
CLIENT_ID = config["WIX"]["CLIENT_ID"]
CLIENT_SECRET = config["WIX"]["CLIENT_SECRET"]

FASTAPI_BEARER_TOKEN = config["FASTAPI"]["BEARER_TOKEN"]
FASTAPI_MIDDLEWARE_SECRECT_KEY = config["FASTAPI"]["MIDDLEWARE_SECRECT_KEY"]

LANGSMITH_API_KEY = config["LANGSMITH"]["API_KEY"]
LANGCHAIN_TRACING_V2 = config["LANGSMITH"]["TRACING_V2"]

os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
