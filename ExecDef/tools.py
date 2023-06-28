import json
import re
from pathlib import Path

from gooddata_pandas import GoodPandas
from gooddata_sdk import Attribute, ExecutionDefinition, GoodDataSdk, ObjId, SimpleMetric


def get_workspace_id() -> str:
    sdk = GoodDataSdk.create_from_profile(profile="default", profiles_path=Path("./profiles.yaml"))
    return sdk.catalog_workspace.list_workspaces()[0].workspace_id


def answer_to_json(answer: str) -> dict:
    """Transform answer to dict, no matter the format.

    This is a bulldozer of a function.

    Args:
        answer (str): Answer from the OpenAI agent.

    Returns:
        dict: Parsed json response
    """
    regex = r"\{(?:[^{}]|)*\}"
    matches = re.search(regex, answer)
    result = matches.group()
    if result == "":
        raise Exception("Did not contain .json!")
    print(result)
    exdef = json.loads(result)
    return exdef


def answer_to_plot(answer: str) -> None:
    """Use Pandas to visualize the generated ExecutionDefinition

    Args:
        answer (str):
            Answer from the OpenAI agent, can be either a valid .json or
            a written text containing the valid .json
    """

    exdef = answer_to_json(answer)

    gp = GoodPandas.create_from_profile(profile="default", profiles_path=Path("./profiles.yaml"))
    frames = gp.data_frames(get_workspace_id())

    exec_def = ExecutionDefinition(
        attributes=[Attribute(local_id=attr, label=attr) for attr in exdef["attributes"]],
        metrics=[SimpleMetric(local_id=metr, item=ObjId(metr, type="metric")) for metr in exdef["metrics"]],
        dimensions=[[attr for attr in exdef["attributes"]], ["measureGroup"]],
        filters=[],
    )

    df, df_metadata = frames.for_exec_def(exec_def=exec_def)

    print(df)
    df.plot.bar()
