"""
Flygbolags-sentiment — steg 7: Implementera och använd modellen
================================================================
En enkel Streamlit-app som visar hur den DistilBERT-modell vi byggde i
notebooken (Airline_Sentiment_Transfer_Learning.ipynb) kan användas i
praktiken av en kundtjänstavdelning:

1. Klassificera enstaka tweets eller en hel fil av tweets på en gång.
2. Se en liten "kundtjänst-dashboard" med de viktigaste insikterna.
3. Läs en kort sammanfattning av hur modellen är byggd.

Kör appen:
    pip install -r requirements.txt
    streamlit run app.py

OBS: Appen förväntar sig att modellen redan är tränad och sparad av
notebooken (steg 6-7 där), dvs filerna:
    airline_sentiment_distilbert.keras
    airline_sentiment_tokenizer/
ska ligga i samma mapp som den här filen (eller ändra sökvägarna nedan).
"""

import os

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Konfiguration
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Flygbolags-sentiment", page_icon="✈️", layout="wide")

MODEL_PATH = "airline_sentiment_distilbert.keras"
TOKENIZER_PATH = "airline_sentiment_tokenizer"
MAX_LEN = 64

DATA_URL = "https://raw.githubusercontent.com/satyajeetkrjha/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv"

LABELS = ["negative", "neutral", "positive"]
LABEL_SV = {"negative": "Negativ", "neutral": "Neutral", "positive": "Positiv"}
COLORS = {"negative": "#D9534F", "neutral": "#F0AD4E", "positive": "#5CB85C"}
NAVY = "#0B3D66"

CUSTOM_CSS = f"""
<style>
.stat-card {{
    background: white; border-radius: 12px; padding: 1.1rem 1.3rem;
    border: 1px solid #E3E9EE; box-shadow: 0 2px 8px rgba(23,35,46,0.06);
}}
.stat-number {{ font-size: 2.1rem; font-weight: 700; color: {NAVY}; }}
.stat-label {{ font-size: 0.85rem; color: #5C6B78; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Modell & tokenizer (cachas så de bara laddas en gång)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Laddar modell...")
def load_model_and_tokenizer():
    if not os.path.exists(MODEL_PATH) or not os.path.isdir(TOKENIZER_PATH):
        return None, None

    import tensorflow as tf
    from transformers import DistilBertTokenizerFast

    model = tf.keras.models.load_model(MODEL_PATH)
    tokenizer = DistilBertTokenizerFast.from_pretrained(TOKENIZER_PATH)
    return model, tokenizer


def predict_batch(texts, model, tokenizer):
    enc = tokenizer(
        list(texts), max_length=MAX_LEN, truncation=True, padding="max_length", return_tensors="tf"
    )
    probs = model.predict([enc["input_ids"], enc["attention_mask"]], verbose=0)
    pred_idx = np.argmax(probs, axis=1)
    return [LABELS[i] for i in pred_idx], probs.max(axis=1)


# ----------------------------------------------------------------------------
# Data för dashboarden (samma dataset som i notebooken)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Hämtar data...")
def load_eda_data():
    df = pd.read_csv(DATA_URL)
    return df[["text", "airline_sentiment", "airline", "negativereason"]].dropna(
        subset=["text", "airline_sentiment"]
    )


# ----------------------------------------------------------------------------
# Sidomeny
# ----------------------------------------------------------------------------
st.sidebar.title("✈️ Flygbolags-sentiment")
st.sidebar.caption("Transfer learning-demo med DistilBERT")
page = st.sidebar.radio(
    "Meny",
    ["🔍 Klassificera tweets", "📊 Kundtjänst-dashboard", "ℹ️ Om modellen"],
)

model, tokenizer = load_model_and_tokenizer()

if model is None:
    st.sidebar.warning(
        "Ingen tränad modell hittades i den här mappen.\n\n"
        "Kör notebooken `Airline_Sentiment_Transfer_Learning.ipynb` i Google Colab "
        "först — den sparar `airline_sentiment_distilbert.keras` och "
        "`airline_sentiment_tokenizer/`. Lägg de filerna i samma mapp som `app.py`."
    )

# ----------------------------------------------------------------------------
# SIDA 1 — Klassificera tweets
# ----------------------------------------------------------------------------
if page == "🔍 Klassificera tweets":
    st.title("Klassificera kundkommentarer")
    st.write(
        "Klistra in en eller flera kommentarer (en per rad) så klassificerar modellen "
        "dem direkt som **negativa**, **neutrala** eller **positiva** — precis som "
        "kundtjänst skulle vilja se det i en triage-vy."
    )

    tab1, tab2 = st.tabs(["Skriv in text", "Ladda upp fil (CSV)"])

    with tab1:
        default_text = (
            "My flight got cancelled and nobody from customer service has responded in hours.\n"
            "Thanks for the amazing service today, my flight left right on time!\n"
            "Flight 245 departs at 14:20 from gate B12."
        )
        text_input = st.text_area("En kommentar per rad:", value=default_text, height=140)
        run = st.button("Klassificera", type="primary", disabled=model is None)

        if run and model is not None:
            lines = [t.strip() for t in text_input.split("\n") if t.strip()]
            if lines:
                labels, confidences = predict_batch(lines, model, tokenizer)
                result_df = pd.DataFrame(
                    {
                        "Text": lines,
                        "Sentiment": [LABEL_SV[l] for l in labels],
                        "Säkerhet": [f"{c:.0%}" for c in confidences],
                        "_label": labels,
                    }
                )

                for _, row in result_df.iterrows():
                    color = COLORS[row["_label"]]
                    st.markdown(
                        f"""<div style="border-left:5px solid {color}; padding:0.6rem 1rem;
                        margin-bottom:0.5rem; background:#F6F9FB; border-radius:6px;">
                        <b style="color:{color}">{row['Sentiment']}</b>
                        <span style="color:#5C6B78; font-size:0.85rem;"> ({row['Säkerhet']} säkerhet)</span>
                        <br>{row['Text']}</div>""",
                        unsafe_allow_html=True,
                    )

    with tab2:
        st.write("Ladda upp en CSV-fil med en kolumn som heter **text**.")
        uploaded = st.file_uploader("Välj fil", type=["csv"])
        if uploaded is not None and model is not None:
            batch_df = pd.read_csv(uploaded)
            if "text" not in batch_df.columns:
                st.error("Filen måste ha en kolumn som heter 'text'.")
            else:
                with st.spinner("Klassificerar..."):
                    labels, confidences = predict_batch(batch_df["text"].astype(str), model, tokenizer)
                batch_df["sentiment"] = labels
                batch_df["säkerhet"] = confidences.round(2)

                c1, c2, c3 = st.columns(3)
                counts = pd.Series(labels).value_counts()
                for col, lab in zip([c1, c2, c3], LABELS):
                    n = int(counts.get(lab, 0))
                    col.markdown(
                        f"""<div class="stat-card">
                        <div class="stat-number" style="color:{COLORS[lab]}">{n}</div>
                        <div class="stat-label">{LABEL_SV[lab]}</div></div>""",
                        unsafe_allow_html=True,
                    )

                st.write("")
                st.dataframe(batch_df, use_container_width=True)
                st.download_button(
                    "⬇️ Ladda ner resultat som CSV",
                    batch_df.to_csv(index=False).encode("utf-8"),
                    file_name="klassificerade_tweets.csv",
                    mime="text/csv",
                )

# ----------------------------------------------------------------------------
# SIDA 2 — Kundtjänst-dashboard (EDA / affärsinsikter)
# ----------------------------------------------------------------------------
elif page == "📊 Kundtjänst-dashboard":
    st.title("Kundtjänst-dashboard")
    st.caption("Baserat på Twitter US Airline Sentiment (Kaggle / Crowdflower), 14 640 tweets")

    try:
        df = load_eda_data()

        total = len(df)
        neg_pct = (df["airline_sentiment"] == "negative").mean() * 100
        pos_pct = (df["airline_sentiment"] == "positive").mean() * 100

        c1, c2, c3 = st.columns(3)
        c1.markdown(
            f"""<div class="stat-card"><div class="stat-number">{total:,}</div>
            <div class="stat-label">Tweets totalt</div></div>""".replace(",", " "),
            unsafe_allow_html=True,
        )
        c2.markdown(
            f"""<div class="stat-card"><div class="stat-number" style="color:{COLORS['negative']}">{neg_pct:.0f}%</div>
            <div class="stat-label">Negativa</div></div>""",
            unsafe_allow_html=True,
        )
        c3.markdown(
            f"""<div class="stat-card"><div class="stat-number" style="color:{COLORS['positive']}">{pos_pct:.0f}%</div>
            <div class="stat-label">Positiva</div></div>""",
            unsafe_allow_html=True,
        )

        st.write("")
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Sentiment per flygbolag")
            tab = (
                df.groupby("airline")["airline_sentiment"]
                .value_counts(normalize=True)
                .unstack()
                .reindex(columns=LABELS)
                * 100
            )
            tab = tab.sort_values("negative", ascending=False)
            st.bar_chart(tab, color=[COLORS[l] for l in LABELS], horizontal=True)

        with col_right:
            st.subheader("Vanligaste klagomålen")
            reasons = df["negativereason"].value_counts().head(8).sort_values()
            st.bar_chart(reasons, horizontal=True, color=COLORS["negative"])

        st.info(
            "💡 **Så här skulle det användas i drift:** koppla modellen till flygbolagets "
            "Twitter-/supportflöde så att varje ny kommentar klassificeras automatiskt. "
            "Negativa kommentarer om t.ex. *Cancelled Flight* eller *Customer Service Issue* "
            "kan då eskaleras direkt till rätt team — istället för att drunkna i flödet."
        )
    except Exception as e:
        st.error(f"Kunde inte hämta data just nu ({e}). Kontrollera internetuppkopplingen.")

# ----------------------------------------------------------------------------
# SIDA 3 — Om modellen
# ----------------------------------------------------------------------------
else:
    st.title("Om modellen")
    st.markdown(
        """
Den här appen använder en **DistilBERT**-modell som byggts med transfer learning
enligt vårt 7-stegsflöde:

1. **Basmodell:** `distilbert-base-uncased` — en destillerad, ~60% snabbare
   version av BERT som redan förstår grammatik och sammanhang i engelsk text.
2. **Data:** Twitter US Airline Sentiment (Kaggle/Crowdflower), 14 640 tweets
   om 6 amerikanska flygbolag, nedsamplat till 3 600 för snabb träning.
3. **Frysta lager:** hela DistilBERT-basen fryst under första träningsfasen.
4. **Egna lager:** `Dense(128, ReLU)` → `Dropout(0.3)` → `Dense(3, Softmax)`
   ovanpå `[CLS]`-representationen.
5. **Fine-tuning:** de två sista transformerblocken låses upp och tränas om
   med en lägre learning rate.
6. **Utvärdering:** accuracy, F1-score och confusion matrix på ett testset
   modellen aldrig sett.
7. **Implementering:** den här Streamlit-appen — steget där modellen faktiskt
   används av en verklig användare (t.ex. kundtjänst).

Se notebooken `Airline_Sentiment_Transfer_Learning.ipynb` och presentationen
för fullständiga detaljer, kod och resultat.
"""
    )
