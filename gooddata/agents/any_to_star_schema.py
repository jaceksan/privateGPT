import os
from time import time
import streamlit as st
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
import openai
from gooddata_sdk import GoodDataSdk
from gooddata_sdk.catalog.data_source.declarative_model.physical_model.pdm import CatalogScanResultPdm
from gooddata.sdk_wrapper import GoodDataSdkWrapper
from gooddata.tools import get_name_for_id

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG")


RESULT_TYPES = {
    "SQL statements": """
Question:
Write a set of SQL statements, each in a separate code snippet.
There should be one DDL SQL statement for each target table.
There should be one DML SQL statement for each target table transferring data from corresponding source tables.
""",
    "dbt models": """
Question:
Write a set of dbt models, each in a separate code snippet.
There should be one dbt model for each target table transferring data from corresponding source tables.
"""
}


def load_chain(model_name: str) -> ConversationChain:
    """Logic for loading the chain you want to use should go here."""
    llm = ChatOpenAI(temperature=0, model_name=model_name)

    chain = ConversationChain(llm=llm)
    return chain


def query_openai(model_name: str, prompt: str):
    return openai.Completion.create(
        model=model_name,
        prompt=prompt,
        temperature=0,
        max_tokens=150,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["#", ";"]
    )


@st.cache_data
def list_data_sources(_sdk: GoodDataSdk):
    return _sdk.catalog_data_source.list_data_sources()


def render_data_source_picker(_sdk: GoodDataSdk):
    data_sources = list_data_sources(_sdk)
    if "data_source_id" not in st.session_state:
        st.session_state["data_source_id"] = data_sources[0].id
    st.selectbox(
        label="Data sources:",
        options=[x.id for x in data_sources],
        format_func=lambda x: get_name_for_id(data_sources, x),
        key="data_source_id",
    )


def render_result_type_picker():
    if "result_type" not in st.session_state:
        st.session_state["result_type"] = "SQL statements"
    st.selectbox(
        label="Result type:",
        options=["SQL statements", "dbt models"],
        key="result_type",
    )


@st.cache_data
def get_supported_models() -> list[str]:
    return [m["id"] for m in openai.Model.list()["data"]]


def render_openai_models_picker():
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo-0613"
    models = get_supported_models()
    st.selectbox(
        label="OpenAI model:",
        options=models,
        key="openai_model",
    )


@st.cache_data
def scan_data_source(_sdk: GoodDataSdk, _logger, data_source_id: str) -> CatalogScanResultPdm:
    _logger.info(f"scan_data_source {data_source_id=} START")
    start = time()
    result = _sdk.catalog_data_source.scan_data_source(data_source_id)
    duration = int((time() - start)*1000)
    _logger.info(f"scan_data_source {data_source_id=} duration={duration}")
    return result


def generate_question_from_model(pdm: CatalogScanResultPdm) -> str:
    result = f"The source tables are:\n"
    # Only 5 tables for demo purposes, large prompt causes slowness
    # TODO - train a LLM model with PDM model once and utilize it
    # for table in pdm.pdm.tables:
    for i in range(5):
        table = pdm.pdm.tables[i]
        column_list = ", ".join([c.name for c in table.columns])
        result += f"- \"{table.path[-1]}\" with columns {column_list}\n"
    return result


def any_to_star_model(sdk: GoodDataSdkWrapper, logger):
    columns = st.columns(3)
    with columns[0]:
        render_data_source_picker(sdk.sdk)
    with columns[1]:
        render_result_type_picker()
    with columns[2]:
        render_openai_models_picker()

    try:
        if st.button("Generate", type="primary"):
            data_source_id = st.session_state["data_source_id"]
            pdm = scan_data_source(sdk.sdk, logger, data_source_id)

            with open("prompts/any_to_star_schema.txt") as fp:
                prompt = fp.read()
            request = prompt + generate_question_from_model(pdm) + RESULT_TYPES[st.session_state.result_type]
            with open("tmp_prompt.txt", "w") as fp:
                fp.write(request)

            logger.info(f"OpenAI query START")
            start = time()
            # output = query_openai(request, st.session_state.openai_model)
            chain = load_chain(st.session_state.openai_model)
            output = chain.run(input=request)
            st.write(output)
            duration = int((time() - start)*1000)
            logger.info(f"OpenAI query duration={duration}")
    except openai.error.AuthenticationError as e:
        st.write("OpenAI unknown authentication error")
        st.write(e.json_body)
        st.write(e.headers)
