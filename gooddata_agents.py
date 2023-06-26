from dotenv import load_dotenv
import streamlit as st
import os
import argparse

from gooddata.logger import get_logger
from gooddata.sdk_wrapper import GoodDataSdkWrapper
from gooddata_sdk import GoodDataSdk
from gooddata.agents.chat import chat_gpt
from gooddata.agents.pandas_dataframe import pandas_df
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
        options=["Data scientist", "Pandas Data Frame"],
        key="agent",
    )


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


if __name__ == "__main__":
    main()

