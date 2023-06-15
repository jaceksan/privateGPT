import os

from sd_converter import Converter

oai_dir = "oai_finetune/"
hf_dir = "hf_finetune/"
src_dir = "source_documents/"

converter = Converter()
for filename in os.listdir(src_dir):
    filepath = os.path.join(src_dir, filename)
    if os.path.isfile(filepath):
        converter.convert(
            input_file=filepath,
            output_file=os.path.join(oai_dir, (filename.rsplit(".", 1)[0] + ".jsonl")),
            format="oai",
        )
        converter.convert(
            input_file=filepath, output_file=os.path.join(hf_dir, (filename.rsplit(".", 1)[0] + ".jsonl")), format="hf"
        )
