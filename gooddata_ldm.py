#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (C) 2023 GoodData Corporation

from gooddata_sdk import GoodDataSdk
import logging
import argparse
import os
import yaml


logging.basicConfig(level=logging.INFO, format="%(levelname)-8s: %(name)s : %(asctime)-15s - %(message)s")


def parse_arguments():
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(
        conflict_handler="resolve",
        description="Generate natural language description for LDM model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-gh", "--gooddata-host",
        help="Hostname(DNS) where GoodData is running",
        default=os.getenv("GOODDATA_HOST", "http://localhost:3000")
    )
    parser.add_argument(
        "-gt", "--gooddata-token",
        help="GoodData API token for authentication",
        default=os.getenv("GOODDATA_TOKEN", "YWRtaW46Ym9vdHN0cmFwOmFkbWluMTIz")
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    logging.info("START")

    sdk = GoodDataSdk.create(host_=args.gooddata_host, token_=args.gooddata_token)

    with open("gooddata_workspaces.yaml") as fp:
        workspaces = yaml.safe_load(fp)

    for workspace_id in workspaces["workspaces"]:
        logging.info(f"Processing workspace {workspace_id=}")
        workspace = sdk.catalog_workspace.get_workspace(workspace_id)
        ldm = sdk.catalog_workspace_content.get_declarative_ldm(workspace_id)
        lines = [
            "This document relates to \"GoodData Tiger platform\".\n\n" +
            f"This document contains GoodData logical data model (LDM) for workspace" +
            f" with ID \"{workspace.id}\" and with title \"{workspace.name}\".\n"
        ]
        with open(f"source_documents/example_ldm_{workspace.id}.txt", "w") as fp:
            lines.append("\nQuestion: what datasets exists in this LDM?\n")
            lines.append("Answer:\n")
            for dataset in ldm.ldm.datasets:
                lines.append(f"Dataset with ID \"{dataset.id}\" has title \"{dataset.title}\".\n")

            for dataset in ldm.ldm.datasets:
                lines.append(f"\nQuestion: what facts are in dataset with ID \"{dataset.id}\"?\n")
                if len(dataset.facts) > 0:
                    lines.append("Answer:\n")
                    lines.append(f"Dataset with ID \"{dataset.id}\" has the following facts:\n")
                    for fact in dataset.facts:
                        lines.append(f"- Fact with ID \"{fact.id}\" has title \"{fact.title}\"\n")
                else:
                    lines.append(f"Dataset with ID \"{dataset.id}\" has no facts.\n")

            for dataset in ldm.ldm.datasets:
                lines.append(f"\nQuestion: what attributes are in dataset with ID \"{dataset.id}\"?\n")
                if len(dataset.attributes) > 0:
                    lines.append("Answer:\n")
                    lines.append(f"Dataset with ID \"{dataset.id}\" has the following attributes:\n")
                    for attribute in dataset.attributes:
                        lines.append(f"- Attribute with ID \"{attribute.id}\" has title \"{attribute.title}\"\n")
                else:
                    lines.append(f"Dataset with ID \"{dataset.id}\" has no attributes.\n")

            dataset = ldm.ldm.datasets[0]
            lines.append(f"\nQuestion: what is the title of the dataset with ID \"{dataset.id}\"?\n")
            lines.append(f"Answer: the title is \"{dataset.title}\"")

            fp.writelines(lines)

    logging.info("END")


if __name__ == "__main__":
    main()
