# GoodData extension for privateGPT

This fork is focused on train LLM with GoodData specific knowledge.

## Training

Generate documents containing GoodData workspace-level metadata - GoodData logical data model.
```shell
python gooddata_ldm.py
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
