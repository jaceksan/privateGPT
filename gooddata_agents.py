from dotenv import load_dotenv
import streamlit as st
import os
import argparse

from gooddata.logger import get_logger
from gooddata.sdk_wrapper import GoodDataSdkWrapper
from gooddata_sdk import GoodDataSdk
from gooddata.agents.chat import chat_gpt
from gooddata.agents.pandas_dataframe import pandas_df
from gooddata.agents.any_to_star_schema import any_to_star_model
from gooddata.agents.report_agent import ReportAgent
from gooddata.tools import get_name_for_id


def parse_arguments():
    parser = argparse.ArgumentParser(
        conflict_handler="resolve",
        description="Talk to GoodData",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Increase logging level to DEBUG')
    parser.add_argument("-gh", "--gooddata-host",
                        help="Hostname(DNS) where GoodData is running",
                        default=os.getenv("GOODDATA_HOST", "http://localhost:3000"))
    parser.add_argument("-gt", "--gooddata-token",
                        help="GoodData API token for authentication",
                        default=os.getenv("GOODDATA_TOKEN", "YWRtaW46Ym9vdHN0cmFwOmFkbWluMTIz"))
    parser.add_argument("-go", "--gooddata-override-host",
                        help="Override hostname, if necessary. "
                             "When you connect to different hostname than where GoodData is running(proxies)",
                        default=os.getenv("GOODDATA_OVERRIDE_HOST"))
    return parser.parse_args()


def render_workspace_picker(sdk: GoodDataSdk):
    workspaces = sdk.catalog_workspace.list_workspaces()
    st.sidebar.selectbox(
        label="Workspaces:",
        options=[w.id for w in workspaces],
        format_func=lambda x: get_name_for_id(workspaces, x),
        key="workspace_id",
    )


def render_agent_picker():
    st.sidebar.selectbox(
        label="Agents:",
        options=["Data scientist", "Pandas Data Frame", "Any to Star Model", "Report executor"],
        key="agent",
    )


def render_chart_type_picker():
    st.selectbox(
        label="Chart type:",
        options=["Table", "Bar chart", "Line chart"],
        key="chart_type",
    )


@st.cache_data
def execute_report(_agent, workspace_id, answer):
    return _agent.execute_report(workspace_id, answer)


def report_executor(workspace_id: str):
    render_chart_type_picker()
    chart_type = st.session_state.get("chart_type")
    agent = ReportAgent(workspace_id)
    query = st.text_area("Enter question:")
    if st.button("Submit Query", type="primary"):
        if query:
            answer = agent.ask(query)
            df, attributes, metrics = execute_report(agent, workspace_id, answer)
            if chart_type == "Bar chart":
                print(df)
                df.set_index(df.columns.values[0], inplace=True)
                # TODO - what the fuck? Looks like a Streamlit bug
                df.index.name = None
                print(df)
                st.bar_chart(df)
            else:
                st.dataframe(df)


def main():
    load_dotenv()
    st.set_page_config(layout="wide", page_icon="favicon.ico", page_title="Talk with your GoodData")
    args = parse_arguments()
    logger = get_logger("streamlit-gooddata", args.debug)
    sdk = GoodDataSdkWrapper(args, logger, timeout=30)

    render_workspace_picker(sdk.sdk)
    render_agent_picker()
    selected_agent = st.session_state.get("agent")
    workspace_id = st.session_state.get("workspace_id")

    if selected_agent == "Data scientist":
        chat_gpt()
    elif selected_agent == "Pandas Data Frame":
        pandas_df(sdk, workspace_id)
    elif selected_agent == "Any to Star Model":
        any_to_star_model(sdk, logger)
    elif selected_agent == "Report executor":
        report_executor(workspace_id)


if __name__ == "__main__":
    main()
