import os
from enum import Enum
from pathlib import Path

import openai
from dotenv import load_dotenv
from gooddata_sdk import GoodDataSdk
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator

load_dotenv()


class Agent:
    class Model(Enum):
        GPT_4 = "gpt-4"
        GPT_4_FUNC = "gpt-4-0613"
        GPT_3 = "gpt-3.5-turbo"
        GPT_3_FUNC = "gpt-3.5-turbo-0613"

    class AIMethod(Enum):
        RAW = "raw"
        FUNC = "functional"
        LANGCHAIN = "langchain"

    def __init__(self, open_ai_model: Model = Model.GPT_4_FUNC, method: AIMethod = AIMethod.FUNC) -> None:
        self.openAIModel = open_ai_model
        self.method = method

        load_dotenv()
        assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY not found in environment variables"
        openai.api_key = os.environ["OPENAI_API_KEY"]

        self.sdk = GoodDataSdk.create_from_profile(profile="default", profiles_path=Path("./profiles.yaml"))
        self.workspace_id = self.sdk.catalog_workspace.list_workspaces()[0].workspace_id
        self.metrics_string = str(
            [
                [(metric.id), metric.title]
                for metric in self.sdk.catalog_workspace_content.get_metrics_catalog(workspace_id=self.workspace_id)
            ]
        )
        self.attribute_string = str(
            [
                [(attr.id), attr.title]
                for attr in self.sdk.catalog_workspace_content.get_attributes_catalog(workspace_id=self.workspace_id)
            ]
        )

    def get_langchain_query(self, question: str) -> str:
        return f"""Create "{question}" from metrics and attributes in AHAHA_workspace as Execution Definition.
            Write only the .json without any explanation.
            This means, that you always start with '{' and end with '}'."""

    def get_open_ai_sys_msg(self) -> str:
        return """
            You create ExecutionDefinition and always write only the .json without any explanation.
            This means, that you always start with "{" and end with "}".

            To create an ExecutionDefinition, you need to provide .json in such structure:
            {
                "attributes": ["string"],
                "metrics": ["string"],
            }

            Where:
            Attributes are a list containing strings, representing the identifier of the attribute from the workspace
            Metrics are a list containing strings, representing the identifier of the metrics from the workspace
            """

    def get_open_ai_fnc_info(self) -> str:
        return f"""
        Whenever you create a new ExecutionDefinition, you always work upon AHAHA_workspace, defined as:
        metrics:{self.metrics_string}
        attributes:{self.attribute_string}
        \"\"\"
        """

    def get_open_ai_raw_prompt(self, question: str) -> str:
        return f"""
        Create "{question}" in AHAHA_workspace as ExecutionDefinition json.

        context:\"\"\"
        This is AHAHA_workspace:
        metrics:{self.metrics_string}
        attributes:{self.attribute_string}
        \"\"\"
        """

    def get_functions_prompt(self, question: str) -> str:
        """Prompt for Function Calls from OpenAI.

        Returns:
            str: OpenAI function call prompt
        """
        return f"""
        Create "{question}" as ExecutionDefinition, strictly from the metrics and attributes in AHAHA_workspace.
        """

    def get_execdef_fnc(self) -> str:
        """ExecutionDefinition function definition for OpenAI function calls

        Returns:
            str: Definition of the ExecDef
        """
        return {
            "name": "ExecutionDefinition",
            "description": "Create ExecutionDefinition for data visualization",
            "parameters": {
                "type": "object",
                "properties": {
                    "attributes": {
                        "type": "array",
                        "items": {"type": "string", "description": "local_id of an attribute"},
                        "description": "List of local_id of attributes to be used in the visualization",
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string", "description": "local_id of a metric"},
                        "description": "List of local_id of metrics to be used in the visualization",
                    },
                },
                "required": ["attributes", "metrics"],
            },
        }

    def ask_open_ai_raw(self, prompt: str) -> str:
        print(
            f"""Asking OpenAI.
              model: {self.openAIModel.value}
              method: {self.method.value}"""
        )

        completion = openai.ChatCompletion.create(
            model=self.openAIModel.value,
            messages=[
                {"role": "system", "content": self.get_open_ai_sys_msg()},
                {"role": "user", "content": self.get_open_ai_raw_prompt(prompt)},
            ],
        )
        print(f"Tokens: {completion.usage}")

        return completion.choices[0].message.content

    def ask_func_open_ai(self, question: str) -> str:
        print(
            f"""Asking OpenAI.
              model: {self.openAIModel.value}
              method: {self.method.value}"""
        )

        completion = openai.ChatCompletion.create(
            model=self.openAIModel.value,
            messages=[
                {"role": "system", "content": self.get_open_ai_fnc_info()},
                {"role": "user", "content": self.get_functions_prompt(question)},
            ],
            functions=[self.get_execdef_fnc()],
            function_call={"name": "ExecutionDefinition"},
        )

        print(f"Tokens: {completion.usage}")
        return completion.choices[0].message.function_call.arguments

    def ask_langchain_open_ai(self, question: str) -> str:
        print(
            f"""Asking OpenAI.
              model: {self.openAIModel.value}
              method: {self.method.value}"""
        )

        loader = DirectoryLoader(Path("./data/"))

        index = VectorstoreIndexCreator().from_loaders([loader])

        chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model=self.openAIModel.value),
            retriever=index.vectorstore.as_retriever(),
        )

        return chain.run(self.get_langchain_query(question))

    def ask(self, question: str) -> str:
        match self.method:
            case self.AIMethod.FUNC:
                return self.ask_func_open_ai(question)
            case self.AIMethod.RAW:
                return self.ask_open_ai_raw(question)
            case self.AIMethod.LANGCHAIN:
                return self.ask_langchain_open_ai(question)

            case _:
                print("No method found, defaulting to RAW")
                return self.ask_open_ai_raw(question)
