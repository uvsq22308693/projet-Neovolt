import streamlit as st
import requests

st.set_page_config(
    page_title="Neovolt - Détection de Fraude",
    page_icon="⚡",
    layout="centered"
)

# =====================
# HEADER
# =====================
st.title("⚡ Neovolt - Détection de Fraude Électrique")
st.markdown(
    "Analyse intelligente des consommations pour détecter les comportements suspects."
)

st.divider()

# =====================
# FORMULAIRE
# =====================
st.subheader("📊 Données du client")

col1, col2 = st.columns(2)

with col1:
    consommation_kwh = st.number_input("Consommation (kWh)", value=120.0)
    conso_j_1 = st.number_input("Conso J-1", value=110.0)
    moyenne_7j = st.number_input("Moyenne 7 jours", value=100.0)
    std_7j = st.number_input("Écart-type 7 jours", value=15.0)
    variation_pct = st.number_input("Variation (%)", value=0.18)
    conso_par_m2 = st.number_input("Conso / m²", value=2.5)

with col2:
    jour_semaine = st.selectbox("Jour de semaine", list(range(7)))
    mois = st.selectbox("Mois", list(range(1, 13)))
    weekend = st.selectbox("Weekend", [0, 1])
    surface_m2 = st.number_input("Surface (m²)", value=50.0)
    nb_personnes_foyer = st.number_input("Nombre de personnes", value=3)
    puissance_souscrite_kva = st.number_input("Puissance (kVA)", value=6.0)

st.divider()

# =====================
# BOUTON ANALYSE
# =====================
if st.button("🔍 Analyser le risque", use_container_width=True):

    payload = {
        "consommation_kwh": consommation_kwh,
        "conso_j_1": conso_j_1,
        "moyenne_7j": moyenne_7j,
        "std_7j": std_7j,
        "variation_pct": variation_pct,
        "conso_par_m2": conso_par_m2,
        "jour_semaine": jour_semaine,
        "mois": mois,
        "weekend": weekend,
        "surface_m2": surface_m2,
        "nb_personnes_foyer": nb_personnes_foyer,
        "puissance_souscrite_kva": puissance_souscrite_kva
    }

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=payload
    )

    result = response.json()

    prediction = result["prediction"]
    proba = result["probabilite_fraude"]

    st.divider()

    # =====================
    # RESULTATS
    # =====================
    st.subheader("📈 Résultat de l'analyse")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Probabilité fraude", f"{proba:.2%}")

    with col2:
        if prediction == 1:
            st.error("🚨 FRAUDE PROBABLE")
        else:
            st.success("✅ CLIENT NORMAL")

    # =====================
    # Jauge de risque
    # =====================
    st.progress(min(proba, 1.0))

    # =====================
    # INTERPRÉTATION
    # =====================
    st.subheader("🧠 Analyse")

    if proba > 0.7:
        st.error("Risque très élevé de fraude ⚠️")
    elif proba > 0.4:
        st.warning("Cas suspect à vérifier")
    else:
        st.success("Comportement normal")