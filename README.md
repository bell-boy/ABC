# ABC design doc

tl;dr: pipeline for finetuning a model to convert images of sheet music into ABC notation. 

# data
data is scraped from [musescore](https://musescore.com/sheetmusic) using a custom scraper. The scraper downloads the sheet music as both ABC and a set of images corresponding to pages

# scraping

The parser is made up of two interconnected worker types:
1. A `Head` worker that is responsible for fetching and enqueuing the URLs of sheet music pages on musescore. 
2. `Scraper` workers that are responsible for dequeuing the URLs, downloading the MusicXML files, and storing them on the local filesystem.
It also captures the number of likes and views each piece of sheet music has, which is used as a rough quality metric for the scraped data.

## usage
Kick off the scraper via 
```bash
uv run scraper.py --num-workers 10 --num-scores 1000 --output-dir /path/to/output/dir
```

The highest quality 10% of the scraped data is used for evaulation, while the remaining 90% is used for training.

# evaluation

We have two core metrics for evaluating the performance of our model:
1. **Levenshtein Distance**: This metric calculates the minimum number of single-character edits (insertions, deletions, substitutions) required to change the predicted ABC notation into the ground truth. 
2. **Negative Log Likelihood (NLL)**: The negative log probablity of the ground truth ABC notation as judged by the model. 0 is the best possible score -- indicating that the model assigned a probability of 1 to the correct output. The worst possible score is positive infinity, which would indicate that the model assigned a probability of 0 to the correct output. 
NLL is useful as it's a continous measure of performance.

# models

All experiments use Qwen2.5 7B-Instruct.

# training

Two training paradigms are explored:
1. **Direct Finetuning**: The model is finetuned directly on input-output pairs
2. **Reinforcement Learning from Verifiable Rewards (RLVR)**: The model is finetuned using a reward signal derived from the evaluation metrics, rather than direct supervision.
