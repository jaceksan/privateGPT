import os
import json
import streamlit as st
from gooddata_sdk import GoodDataSdk
from gooddata.sdk_wrapper import GoodDataSdkWrapper
from gooddata.tools import get_title_for_id
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = "org-FGVSBhEOLC3mgOJhR0kXIO1n"


def ask_agent(agent, query):
    """
    Query an agent and return the response as a string.

    Args:
        agent: The agent to query.
        query: The query to ask the agent.

    Returns:ch
        The response from the agent as a string.
    """
    # Prepare the prompt with query guidelines and formatting
    prompt = (
            """
            Let's decode the way to respond to the queries. The responses depend on the type of information requested in the query. 
    
            1. If the query requires a table, format your answer like this:
               {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}
    
            2. For a bar chart, respond like this:
               {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}
    
            3. If a line chart is more appropriate, your reply should look like this:
               {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}
    
            Note: We only accommodate two types of charts: "bar" and "line".
    
            4. For a plain question that doesn't need a chart or table, your response should be:
               {"answer": "Your answer goes here"}
    
            For example:
               {"answer": "The Product with the highest Orders is '15143Exfo'"}
    
            5. If the answer is not known or available, respond with:
               {"answer": "I do not know."}
    
            Return all output as a string. Remember to encase all strings in the "columns" list and data list in double quotes. 
            For example: {"columns": ["Products", "Orders"], "data": [["51993Masc", 191], ["49631Foun", 152]]}
    
            Now, let's tackle the query step by step. Here's the query for you to work on: 
            """
            + query
    )

    # Run the prompt through the agent and capture the response.
    response = agent.run(prompt)

    # Return the response converted to a string.
    return str(response)


def write_answer(response_dict: dict):
    """
    Write a response from an agent to a Streamlit app.

    Args:
        response_dict: The response from the agent.

    Returns:
        None.
    """

    # Check if the response is an answer.
    if "answer" in response_dict:
        st.write(response_dict["answer"])

    # Check if the response is a bar chart.
    # Check if the response is a bar chart.
    if "bar" in response_dict:
        data = response_dict["bar"]
        try:
            df_data = {
                col: [x[i] if isinstance(x, list) else x for x in data['data']]
                for i, col in enumerate(data['columns'])
            }
            df = pd.DataFrame(df_data)
            df.set_index("Products", inplace=True)
            st.bar_chart(df)
        except ValueError:
            print(f"Couldn't create DataFrame from data: {data}")

    # Check if the response is a line chart.
    if "line" in response_dict:
        data = response_dict["line"]
        try:
            df_data = {col: [x[i] for x in data['data']] for i, col in enumerate(data['columns'])}
            df = pd.DataFrame(df_data)
            df.set_index("Products", inplace=True)
            st.line_chart(df)
        except ValueError:
            print(f"Couldn't create DataFrame from data: {data}")

    # Check if the response is a table.
    if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)


def render_insight_picker(sdk: GoodDataSdk, workspace_id: str):
    insights = sdk.insights.get_insights(workspace_id)
    st.selectbox(
        label="Insights:",
        options=[w.id for w in insights],
        format_func=lambda x: get_title_for_id(insights, x),
        key="insight_id",
    )


@st.cache_data
def execute_insight(_sdk: GoodDataSdkWrapper, workspace_id: str, insight_id: str) -> pd.DataFrame:
    return _sdk.pandas.data_frames(workspace_id).for_insight(insight_id)


def pandas_df(sdk: GoodDataSdkWrapper, workspace_id: str):
    if "insight_id" not in st.session_state:
        st.session_state["insight_id"] = None

    render_insight_picker(sdk.sdk, workspace_id)
    insight_id = st.session_state["insight_id"]
    if insight_id:
        df = execute_insight(sdk, workspace_id, insight_id)
        st.dataframe(df)

        query = st.text_area("Enter question about this insight:")
        if st.button("Submit Query", type="primary"):
            if query:
                agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
                response = ask_agent(agent=agent, query=query)
                write_answer(json.loads(response))
