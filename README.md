# Sentimentanalys av flygbolagstweets — Transfer Learning med DistilBERT

Ett litet, snabbt transfer learning-projekt: kan en förtränad språkmodell (DistilBERT)
läras att automatiskt klassificera kundkommentarer om flygbolag på sociala medier —
både *vad* kunden känner (sentiment) och *varför* (orsak)?

**Dataset:** [Twitter US Airline Sentiment](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)
(Kaggle / Crowdflower), 14 640 tweets om 6 amerikanska flygbolag.

## Struktur

Projektet är uppdelat i 6 notebooks som körs i ordning — varje notebook gör en sak och
sparar resultatet till disk så nästa kan bygga vidare:

| # | Notebook | Vad den gör |
|---|----------|-------------|
| 01 | [`01_data_preparation.ipynb`](notebooks/01_data_preparation.ipynb) | Hämtar, städar och delar upp datan |
| 02 | [`02_eda.ipynb`](notebooks/02_eda.ipynb) | Utforskar datan och bygger databerättelsen |
| 03 | [`03_unsupervised_clustering.ipynb`](notebooks/03_unsupervised_clustering.ipynb) | Undersöker om förtränade BERT-embeddings själva grupperar tweets efter sentiment — innan någon träning |
| 04 | [`04_supervised_classification.ipynb`](notebooks/04_supervised_classification.ipynb) | Transfer learning: tränar DistilBERT att klassificera sentiment |
| 05 | [`05_negative_reason_classification.ipynb`](notebooks/05_negative_reason_classification.ipynb) | Transfer learning på en andra uppgift: klassificera varför en tweet är negativ |
| 06 | [`06_evaluation.ipynb`](notebooks/06_evaluation.ipynb) | Utvärderar båda modellerna och sammanfattar resultatet |

```
.
├── notebooks/        01–06, körs i ordning i Google Colab
├── charts/           EDA-diagram (genereras/uppdateras av 02, 03, 04, 05, 06)
├── data/             (gitignored) skapas av 01
├── models/           (gitignored) skapas av 04 och 05
├── app/              Streamlit-app: steg 7, använda modellerna i praktiken
└── presentation/     Bildspel för projektpresentationen (~15 min)
```

## Kom igång

1. Öppna notebook 01 i **Google Colab** (Runtime → Change runtime type → GPU för 03–06).
2. Kör `01` → `02` → `03` → `04` → `05` → `06` **i samma Colab-session**, i ordning.
   Varje notebook läser filer som en tidigare notebook sparat till `./data/`, `./charts/`
   eller `./models/` — öppnar du en ny runtime mellan notebooks försvinner filerna.
3. Ladda ner de sparade modellfilerna från Colab till `app/` och kör Streamlit-appen lokalt:
   ```bash
   cd app
   pip install -r requirements.txt
   streamlit run app.py
   ```

## Metod i korthet

- **Basmodell:** `distilbert-base-uncased` — ~60% snabbare än BERT, behåller ~97% av
  språkförståelsen.
- **Transfer learning-flöde:** frys bas → lägg till egna lager (Dense → Dropout → Dense) →
  träna huvudet → lås upp de två sista transformerblocken → fine-tuna med lägre learning rate.
- **Två modeller:** sentiment (negative/neutral/positive) och orsak till negativt sentiment
  (6 kategorier), tränade med samma recept.

Se [`presentation/`](presentation/) för problembeskrivning, metod, databerättelse och
affärsvärde i sin helhet.
