# GoodData extension for privateGPT

This fork is focused on train LLM with GoodData specific knowledge.

## Training

Generate documents containing GoodData workspace-level metadata - GoodData logical data model.
```shell
python gooddata_ldm.py
```

## Usage
Check example_questions/gooddata.txt for example questions.

The main script privateGPT.py is extended accordingly, prompting LLM with GoodData specific needs.

There are new arguments available in privateGPT.py:
```shell
python privateGPT.py -p 16 -w demo
```

`-p 16` - force privateGPT to use 16 threads to utilize all HW resources
`-w demo` - force privateGPT to work only in the context of workspace with ID=demo
