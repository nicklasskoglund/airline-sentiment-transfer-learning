# Airline Tweet Sentiment Analysis — Transfer Learning with DistilBERT

A small, fast transfer learning project: can a pretrained language model (DistilBERT)
be taught to automatically classify customer comments about airlines on social media —
both *what* the customer feels (sentiment) and *why* (reason)?

**Dataset:** [Twitter US Airline Sentiment](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)
(Kaggle / Crowdflower), 14,640 tweets about 6 US airlines.

## Structure

The project is split into 6 notebooks that run in order — each notebook does one thing
and saves its result to disk so the next one can build on it:

| # | Notebook | What it does |
|---|----------|---------------|
| 01 | [`01_data_preparation.ipynb`](notebooks/01_data_preparation.ipynb) | Fetches, cleans, and splits the data |
| 02 | [`02_eda.ipynb`](notebooks/02_eda.ipynb) | Explores the data and builds the data story |
| 03 | [`03_unsupervised_clustering.ipynb`](notebooks/03_unsupervised_clustering.ipynb) | Checks whether pretrained BERT embeddings already group tweets by sentiment — before any training |
| 04 | [`04_supervised_classification.ipynb`](notebooks/04_supervised_classification.ipynb) | Transfer learning: trains DistilBERT to classify sentiment |
| 05 | [`05_negative_reason_classification.ipynb`](notebooks/05_negative_reason_classification.ipynb) | Transfer learning on a second task: classifying why a tweet is negative |
| 06 | [`06_evaluation.ipynb`](notebooks/06_evaluation.ipynb) | Evaluates both models and summarizes the results |

    .
    ├── notebooks/        01–06, run in order in Google Colab
    ├── charts/           EDA charts (generated/updated by 02, 03, 04, 05, 06)
    ├── data/             (gitignored) created by 01
    ├── models/           (gitignored) created by 04 and 05
    ├── app/              Streamlit app: step 7, using the models in practice
    └── presentation/     Slide deck for the project presentation (~15 min)

## Getting started

1. Open notebook 01 in **Google Colab** (Runtime → Change runtime type → GPU for 03–06).
2. Run `01` → `02` → `03` → `04` → `05` → `06` **in the same Colab session**, in order.
   Each notebook reads files a previous notebook saved to `./data/`, `./charts/`, or
   `./models/` — opening a new runtime between notebooks means those files are lost.
3. Download the saved model files from Colab into `app/` and run the Streamlit app locally:

       cd app
       pip install -r requirements.txt
       streamlit run app.py

## Method in brief

- **Base model:** `distilbert-base-uncased` — ~60% faster than BERT, retains ~97% of its
  language understanding.
- **Transfer learning flow:** freeze base → add custom layers (Dense → Dropout → Dense) →
  train the head → unfreeze the last two transformer blocks → fine-tune with a lower
  learning rate.
- **Two models:** sentiment (negative/neutral/positive) and reason behind negative
  sentiment (6 categories), trained with the same recipe.

See [`presentation/`](presentation/) for the full problem description, method, data
story, and business value.