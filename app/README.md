# Airline Sentiment — Streamlit App (Step 7: Deploy the Model)

This app is the final step of the transfer learning flow: actually **using** the
DistilBERT model trained in `04_supervised_classification.ipynb`.

## Running the app

1. **Run the notebooks first** (in Google Colab — see `notebooks/01_data_preparation.ipynb`
   for setup instructions, then run `01` → `04` in order). Notebook 04 saves:
   - `models/sentiment_model.keras`
   - `models/sentiment_tokenizer/`

2. Download those files from Colab and place them in the `app/models/` folder
   (or update `MODEL_PATH` / `TOKENIZER_PATH` at the top of `app.py` to match
   wherever you put them).

3. Install dependencies:
```bash
   pip install -r requirements.txt
```

4. Start the app:
```bash
   streamlit run app.py
```

5. Open the link shown in the terminal (usually `http://localhost:8501`).

## What the app includes

- **🔍 Classify tweets** — type text directly or upload a CSV (column `text`) to
  classify several comments at once, with a downloadable result.
- **📊 Customer service dashboard** — the same business insights as in the
  presentation (sentiment distribution, by airline, most common complaints),
  fetched live from the dataset.
- **ℹ️ About the model** — a short summary of how the model is built.

> If the model files are missing, the app shows a clear message in the sidebar
> instead of crashing — the dashboard and info pages still work without a
> trained model.