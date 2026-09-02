# Flygbolags-sentiment — Streamlit-app (Steg 7: Implementera modellen)

Den här appen är det sista steget i transfer learning-flödet: att faktiskt
**använda** den DistilBERT-modell som tränas i
`Airline_Sentiment_Transfer_Learning.ipynb`.

## Så kör du appen

1. **Kör notebooken först** (i Google Colab, se instruktionerna högst upp i den).
   Sista cellerna sparar två saker:
   - `airline_sentiment_distilbert.keras`
   - mappen `airline_sentiment_tokenizer/`

2. Ladda ner de filerna från Colab och lägg dem i **samma mapp** som `app.py`.

3. Installera beroenden:
   ```bash
   pip install -r requirements.txt
   ```

4. Starta appen:
   ```bash
   streamlit run app.py
   ```

5. Öppna länken som visas i terminalen (vanligtvis `http://localhost:8501`).

## Vad appen innehåller

- **🔍 Klassificera tweets** — skriv in text direkt eller ladda upp en CSV
  (kolumn `text`) för att klassificera flera kommentarer på en gång, med
  nedladdningsbart resultat.
- **📊 Kundtjänst-dashboard** — samma affärsinsikter som i presentationen
  (sentimentfördelning, per flygbolag, vanligaste klagomål), hämtat live
  från datasetet.
- **ℹ️ Om modellen** — kort sammanfattning av hur modellen är byggd.

> Om modellfilerna saknas visar appen ett tydligt meddelande om detta i
> sidomenyn istället för att krascha — dashboard- och info-sidorna fungerar
> även utan tränad modell.
