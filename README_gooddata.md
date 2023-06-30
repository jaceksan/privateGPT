# GoodData extension for privateGPT

This fork is focused on train LLM with GoodData specific knowledge.

## Training

Generate documents containing GoodData workspace-level metadata - GoodData logical data model.
```shell
python gooddata_ldm.py
```

### Convert training data

To convert the training data, use the `python SD_converter.py` script. The script has three arguments: <input_file> <output_file> <format> to learn more about it, use the `-h` option.

Exemplary usage:
```shell
python SD_converter.py source_documents/gooddata_ldm.txt openAI-finetune/gooddata.ldm.jsonl Jsonl
```


## Usage
Check example_questions/*.txt for example questions for each role.

The main script privateGPT.py is extended accordingly, prompting LLM in various roles.

There are new arguments available in privateGPT.py:
```shell
python privateGPT.py -p 8 -w demo -r gooddata
```

- -p
  - force privateGPT to use 8 threads to utilize all HW resources
  - be careful. E.g. t14s Ryzen 7 has 16 threads, but the best performance is with setting 8 threads.
- -w
  - force privateGPT to work only in the context of workspace with ID=demo
- -r
  - Specify role.
  - File `prompts/<role>.txt` is injected into each question as prompt.
  - If needed, variables are injected, e.g. workspace_id.

# GoodData's agents powered by LangChain

Connected to OpenAI API, put your OPENAI_API_KEY and OPENAPI_ORG to .env file.

Run:
```shell
streamlit run gooddata_agents.py
```

and play!

TODO: 
finish the ultimate use case - based on user query, setup GD execution definition, execute report and explain result. 
