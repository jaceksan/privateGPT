# Guide how to fine-tune

Official documentation [link](https://platform.openai.com/docs/guides/fine-tuning).

## Prerequisites
* `pip install --upgrade openai`
* `export OPENAI_API_KEY=<API-KEY>`
* prepare your train dataset based on the [best practise guide](https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset)

## Commands using CLI

### List fine-tuned models
```bash
openai api fine_tunes.list

#{
#  "object": "list",
#  "data": []
#}
```

### List available models
```bash
openai api models.list
```

### Create fine-tuned model from base model
```bash
openai api fine_tunes.create -t test.jsonl -m ada --suffix "custom model name"

# resulting name of the model will be
# ada:ft-your-org:custom-model-name-2022-02-15-04-21-04
```
[Useful link](https://platform.openai.com/docs/guides/fine-tuning/create-a-fine-tuned-model)

### Get result from fine-tuning
```bash
openai api fine_tunes.results -i <YOUR_FINE_TUNE_JOB_ID>

# use list fine-tuned models to obtain job id
```

### Fine-tuned already fine-tuned model
Read the following [link](https://platform.openai.com/docs/guides/fine-tuning/continue-fine-tuning-from-a-fine-tuned-model) how to do that.


### Fine-tuned model usage

Using CLI
```bash
openai api completions.create -m <FINE_TUNED_MODEL> -p <YOUR_PROMPT>
```

Using API
```bash
curl https://api.openai.com/v1/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt": YOUR_PROMPT, "model": FINE_TUNED_MODEL}'
```

Python
```python
import openai
openai.Completion.create(
model=FINE_TUNED_MODEL,
prompt=YOUR_PROMPT)
```

[Useful link](https://platform.openai.com/docs/guides/fine-tuning/use-a-fine-tuned-model)
