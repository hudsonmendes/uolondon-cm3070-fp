# Package `hlm12erc`

Hi, my name is Hudson Leonardo MENDES!

This python package contains all the software produced for my<br />
CM3070 Final Computer Science Project by the Unviersity of London<br />
for my BSc Computer Science specialised in Machine Learning & Data Science.

This `README.md` should contain everything that is required to<br />
run not just my Algorithm/Model for Emotion Recognition in Conversations<br />
but also all other utility code that I have used to work on the project,<br />
including, but not limited to:

- ETL (transforming the raw MELD data into 1NF)
- Arxiv API searches (used as data to back some conclusions)
- ERC Inference itself

## Data

The following code/commands have been used to fetch data from<br />
data sources that was required for the investigation carried out<br />
during the project

### Arxiv Natural Language Paper Count per Term

During the literature review, it has become clear that Transformers was<br />
seen by multiple authors as the successor of RNN-based embedding models.<br />
The search below was used to collect the popularity of several terms<br />
related to textual feature extraction, to see whether this is also<br />
demonstrated by popularity in research papers.

```
export ARXIV_PAPERS_PATH="/tmp/arxiv-papers-results.txt"
rm -rf $ARXIV_PAPERS_PATH

python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2017 --term "tf-idf"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2017 --term "bert"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2017 --term "distilbert"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2017 --term "roberta"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2017 --term "distilroberta"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2017 --term "gpt"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2017 --term "gpt-2"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2017 --term "gpt-3"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2017 --term "elmo"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2017 --term "rnn"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2017 --term "gru"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2017 --term "lstm"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2017 --term "bilstm"

python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2018 --term "tf-idf"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2018 --term "bert"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2018 --term "distilbert"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2018 --term "roberta"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2018 --term "distilroberta"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2018 --term "gpt"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2018 --term "gpt-2"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2018 --term "gpt-3"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2018 --term "elmo"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2018 --term "rnn"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2018 --term "gru"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2018 --term "lstm"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2018 --term "bilstm"

python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2019 --term "tf-idf"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2019 --term "bert"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2019 --term "distilbert"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2019 --term "roberta"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2019 --term "distilroberta"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2019 --term "gpt"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2019 --term "gpt-2"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2019 --term "gpt-3"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2019 --term "elmo"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2019 --term "rnn"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2019 --term "gru"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2019 --term "lstm"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2019 --term "bilstm"

python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2020 --term "tf-idf"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2020 --term "bert"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2020 --term "distilbert"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2020 --term "roberta"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2020 --term "distilroberta"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2020 --term "gpt"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2020 --term "gpt-2"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2020 --term "gpt-3"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2020 --term "elmo"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2020 --term "rnn"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2020 --term "gru"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2020 --term "lstm"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2020 --term "bilstm"

python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2021 --term "tf-idf"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2021 --term "bert"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2021 --term "distilbert"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2021 --term "roberta"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2021 --term "distilroberta"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2021 --term "gpt"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2021 --term "gpt-2"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2021 --term "gpt-3"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2021 --term "elmo"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2021 --term "rnn"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2021 --term "gru"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2021 --term "lstm"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2021 --term "bilstm"

python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2022 --term "tf-idf"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2022 --term "bert"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2022 --term "distilbert"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2022 --term "roberta"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2022 --term "distilroberta"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2022 --term "gpt"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2022 --term "gpt-2"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2022 --term "gpt-3"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2022 --term "elmo"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2022 --term "rnn"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2022 --term "gru"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2022 --term "lstm"
python -m hlm12erc arxiv count --out $ARXIV_PAPERS_PATH --year 2022 --term "bilstm"
```
