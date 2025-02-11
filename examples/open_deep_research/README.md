# Open Deep Research

Welcome to this open replication of [OpenAI's Deep Research](https://openai.com/index/introducing-deep-research/)!

Read more about this implementation's goal and methods [in our blog post](https://huggingface.co/blog/open-deep-research).

This agent achieves 55% pass@1 on GAIA validation set, vs 67% for Deep Research.

To install it, first run
```bash
pip install -r requirements.txt
```

And install smolagents dev version
```bash
pip install -e .[dev]
```

Then you're good to go! Run the run.py script, as in:
```bash
python run.py --model-id "o1" --question "Your question here!"
```

## Full reproducibility of results
> [!WARNING]
> The data used in our submissions to GAIA was augmented in this way:
> -  For each single-page .pdf or .xls file, it was opened in a file reader (MacOS Sonoma Numbers or Preview), and a ".png" screenshot was taken and added to the folder.
> - Then for any file used in a question, the file loading system checks if there is a ".png" extension version of the file, and loads it instead of the original if it exists.
> This process was done manually but could be automatized.
