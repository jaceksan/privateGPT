import json
import os
from typing import List, Tuple


class Converter:
    """A class to represent a converter for AI fine-tuning notations."""

    def __init__(self):
        return

    def _to_file(self, output_file: str, content: List[str]) -> None:
        """
        Writes the content to a target file.

        Args:
            output_file (str): Target file.
            content (List[str]): Content, usually .jsonl format written in lines.
        """
        with open(output_file, "w") as f:
            for line in content:
                f.write(f"{line}\n")

    def _parse_input_file(self, input_file: str) -> List[Tuple[str, str]]:
        """
        Reads and parses the input file into a list of Question-Answer pairs.

        Args:
            input_file (str): Path to the input file.

        Returns:
            List[Tuple[str, str]]: A list of Question-Answer pairs.
        """

        try:
            with open(input_file, "r") as file:
                text = file.read()
            entries = text.split("Question:")

            if len(entries) == 1:
                raise ValueError(f"No 'Question:' found in the input file: {input_file}")

            qa_pairs = []
            for entry in entries[1:]:
                try:
                    question, answer = entry.split("Answer:")
                except ValueError:
                    raise ValueError(f"Missing 'Answer:' keyword in the input file: {input_file}")
                question = question.strip()
                answer = answer.strip()
                qa_pairs.append((answer, question))

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {input_file}")
        except Exception:
            raise Exception(f"Invalid file format: {input_file}")

        return qa_pairs

    def _to_oai(self, qa_pairs: List[Tuple[str, str]]) -> List[str]:
        """
        Converts a list of Question-Answer pairs to the JSONL format and writes to the output file.

        Args:
            qa_pairs (List[Tuple[str, str]]): A list of Question-Answer pairs.

        Returns:
            List[str]: Content of the final .jsonl file
        """

        content = []
        for answer, question in qa_pairs:
            qa_pair = {"question": question, "answer": answer}
            content.append(json.dumps(qa_pair))
        return content

    def _to_hf(self, qa_pairs: List[Tuple[str, str]]) -> List[str]:
        """
        Converts a list of Question-Answer pairs to the Choices format and writes to the output file.

        Args:
            qa_pairs (List[Tuple[str, str]]): A list of Question-Answer pairs.

        Returns:
            List[str]: Content of the final .jsonl file
        """
        content = []
        for answer, question in qa_pairs:
            # Choices is a list, hence the use of square brackets ([]).
            qa_pair = {"Question": question, "choices": [answer]}
            content.append(json.dumps(qa_pair))
        return content

    def convert(self, input_file: str, output_file: str, format: str) -> None:
        """
        Converts the input file to the specified format and writes to the output file.

        Args:
            input_file (str): Path to the input file.
            output_file (str): Path to the output file.
            format (str): Conversion format (oai or hf).
        """
        if not os.path.isfile(input_file):
            raise Exception("Input file does not exist!")

        if os.path.isfile(output_file):
            print(f"Output file: {output_file} exists. Overwriting.")

        qa_pairs = self._parse_input_file(input_file)

        match format:
            case "oai":
                self._to_file(output_file=output_file, content=self._to_oai(qa_pairs=qa_pairs))
            case "hf":
                self._to_file(output_file=output_file, content=self._to_hf(qa_pairs=qa_pairs))
            case _:
                raise Exception("Wrong format argument")


if __name__ == "__main__":
    """The main entry point of the script. Runs when the script is called from the command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="This is a helper script to convert between different AI fine-tuning notations."
    )
    parser.add_argument("-i", "--input", type=str, help="Path to input file")
    parser.add_argument("-o", "--output", type=str, help="Path to output file", default="output.jsonl")
    parser.add_argument(
        "-f", "--format", type=str, choices=["oai", "hf"], default="oai", help="Output format (oai or hf)"
    )

    args = parser.parse_args()

    converter = Converter()
    converter.convert(input_file=args.input, output_file=args.output, format=args.format)
