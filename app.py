from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from matplotlib.ticker import FuncFormatter

load_dotenv()

try:
    from google import genai
except ImportError:
    genai = None


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "kc_house_data.csv"
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def format_currency(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"${value:,.0f}"


def format_currency_delta(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    sign = "+" if value > 0 else "-" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


def format_number(value: float, decimals: int = 0) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def format_percent(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.1%}"


def dollar_axis_formatter(value: float, _: int) -> str:
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.0f}k"
    return f"${value:,.0f}"


def get_api_key() -> str:
    try:
        secret_value = st.secrets.get("GOOGLE_API_KEY", "") or st.secrets.get(
            "GEMINI_API_KEY", ""
        )
    except Exception:
        secret_value = ""

    return secret_value or os.getenv("GOOGLE_API_KEY", "") or os.getenv(
        "GEMINI_API_KEY", ""
    )


def dataset_signature(df: pd.DataFrame) -> str:
    if df.empty:
        return "empty"
    ids_hash = int(pd.util.hash_pandas_object(df["id"], index=False).sum())
    return f"{len(df)}-{int(df['price'].sum())}-{ids_hash}"


def property_signature(selected_property: pd.Series, comparables: pd.DataFrame) -> str:
    if comparables.empty:
        return str(selected_property["id"])
    comp_hash = int(pd.util.hash_pandas_object(comparables["id"], index=False).sum())
    return f"{selected_property['id']}-{len(comparables)}-{comp_hash}"


def classify_valuation(gap_pct: float) -> str:
    if pd.isna(gap_pct):
        return "Indeterminee"
    if gap_pct <= -0.05:
        return "Sous-evaluee"
    if gap_pct >= 0.05:
        return "Surevaluee"
    return "Bien positionnee"


def show_figure(fig: plt.Figure) -> None:
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    numeric_columns = [
        "price",
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "view",
        "condition",
        "grade",
        "sqft_above",
        "sqft_basement",
        "yr_built",
        "yr_renovated",
        "lat",
        "long",
        "sqft_living15",
        "sqft_lot15",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["id"] = df["id"].astype(str)
    df["zipcode"] = df["zipcode"].astype(str)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%S", errors="coerce")
    df["sale_year"] = df["date"].dt.year
    df["price_per_sqft"] = np.where(
        df["sqft_living"].gt(0),
        df["price"] / df["sqft_living"],
        np.nan,
    )
    df["age"] = df["sale_year"] - df["yr_built"]
    df["is_renovated"] = df["yr_renovated"].fillna(0).gt(0)
    df["has_basement"] = df["sqft_basement"].fillna(0).gt(0)
    df["waterfront_label"] = np.where(df["waterfront"].fillna(0).eq(1), "Oui", "Non")

    return df.sort_values("date", ascending=False).reset_index(drop=True)


def render_sidebar_filters(df: pd.DataFrame) -> dict:
    st.sidebar.header("Filtres du marche")
    st.sidebar.caption("Les filtres ci-dessous s'appliquent aux deux onglets.")

    bedrooms_min = int(df["bedrooms"].min())
    bedrooms_max = int(df["bedrooms"].max())
    grade_min = int(df["grade"].min())
    grade_max = int(df["grade"].max())
    year_min = int(df["yr_built"].min())
    year_max = int(df["yr_built"].max())

    bathrooms_min = float(df["bathrooms"].min())
    bathrooms_max = float(df["bathrooms"].max())

    filters = {
        "bedrooms": st.sidebar.slider(
            "Nombre de chambres",
            min_value=bedrooms_min,
            max_value=bedrooms_max,
            value=(bedrooms_min, bedrooms_max),
        ),
        "bathrooms": st.sidebar.slider(
            "Nombre de salles de bains",
            min_value=bathrooms_min,
            max_value=bathrooms_max,
            value=(bathrooms_min, bathrooms_max),
            step=0.25,
        ),
        "zipcodes": st.sidebar.multiselect(
            "Codes postaux",
            options=sorted(df["zipcode"].unique()),
        ),
        "grade": st.sidebar.slider(
            "Qualite de construction (grade)",
            min_value=grade_min,
            max_value=grade_max,
            value=(grade_min, grade_max),
        ),
        "waterfront_only": st.sidebar.checkbox(
            "Seulement les proprietes avec front de mer",
            value=False,
        ),
        "yr_built": st.sidebar.slider(
            "Annee de construction",
            min_value=year_min,
            max_value=year_max,
            value=(year_min, year_max),
        ),
    }

    if get_api_key():
        st.sidebar.success("Cle Google detectee pour les analyses LLM.")
    else:
        st.sidebar.warning(
            "Ajoutez GOOGLE_API_KEY dans .env ou dans les secrets Streamlit "
            "pour activer Gemini."
        )

    return filters


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    filtered_df = df[
        df["bedrooms"].between(*filters["bedrooms"])
        & df["bathrooms"].between(*filters["bathrooms"])
        & df["grade"].between(*filters["grade"])
        & df["yr_built"].between(*filters["yr_built"])
    ].copy()

    if filters["zipcodes"]:
        filtered_df = filtered_df[filtered_df["zipcode"].isin(filters["zipcodes"])]
    if filters["waterfront_only"]:
        filtered_df = filtered_df[filtered_df["waterfront"].eq(1)]

    return filtered_df.sort_values("date", ascending=False).reset_index(drop=True)


def plot_price_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df["price"].dropna(), bins=30, color="#1f77b4", edgecolor="white")
    ax.set_title("Distribution des prix")
    ax.set_xlabel("Prix de vente (USD)")
    ax.set_ylabel("Nombre de proprietes")
    ax.xaxis.set_major_formatter(FuncFormatter(dollar_axis_formatter))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_surface_vs_price(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    scatter = ax.scatter(
        df["sqft_living"],
        df["price"],
        c=df["grade"],
        cmap="viridis",
        alpha=0.7,
        s=35,
    )
    ax.set_title("Prix vs superficie habitable")
    ax.set_xlabel("Superficie habitable (pi2)")
    ax.set_ylabel("Prix de vente (USD)")
    ax.yaxis.set_major_formatter(FuncFormatter(dollar_axis_formatter))
    ax.grid(alpha=0.2)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Grade")
    fig.tight_layout()
    return fig


def plot_avg_price_by_bedrooms(df: pd.DataFrame) -> plt.Figure:
    grouped = (
        df.dropna(subset=["bedrooms"])
        .groupby("bedrooms", as_index=False)["price"]
        .mean()
        .sort_values("bedrooms")
    )
    if len(grouped) > 10:
        grouped = grouped.head(10)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        grouped["bedrooms"].astype(int).astype(str),
        grouped["price"],
        color="#16a085",
    )
    ax.set_title("Prix moyen par nombre de chambres")
    ax.set_xlabel("Nombre de chambres")
    ax.set_ylabel("Prix moyen (USD)")
    ax.yaxis.set_major_formatter(FuncFormatter(dollar_axis_formatter))
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def build_market_prompt(filtered_df: pd.DataFrame) -> str:
    top_zipcodes = (
        filtered_df["zipcode"].value_counts().head(5).to_dict()
        if not filtered_df.empty
        else {}
    )

    return f"""
Tu es un analyste immobilier senior. Produis une synthese en francais du marche
residentiel filtre du comte de King.

Contexte quantitatif:
- Nombre de proprietes: {len(filtered_df)}
- Prix moyen: {format_currency(filtered_df['price'].mean())}
- Prix median: {format_currency(filtered_df['price'].median())}
- Prix minimum: {format_currency(filtered_df['price'].min())}
- Prix maximum: {format_currency(filtered_df['price'].max())}
- Prix moyen au pi2: {format_currency(filtered_df['price_per_sqft'].mean())}
- Superficie moyenne: {format_number(filtered_df['sqft_living'].mean(), 0)} pi2
- Age moyen du parc: {format_number(filtered_df['age'].mean(), 1)} ans
- Part renovee: {format_percent(filtered_df['is_renovated'].mean())}
- Top zipcodes: {top_zipcodes}

Instruction:
- Rends la reponse concise et professionnelle.
- Structure la reponse en 3 parties: tendance des prix, profil des biens,
  opportunites ou risques.
- Termine par une recommandation actionnable pour un investisseur junior.
""".strip()


def build_property_prompt(
    selected_property: pd.Series,
    comparables: pd.DataFrame,
    rule_label: str,
) -> str:
    avg_comp_price = comparables["price"].mean()
    gap_value = selected_property["price"] - avg_comp_price
    gap_pct = gap_value / avg_comp_price if avg_comp_price else np.nan

    comparable_lines = []
    for row in comparables.head(5).itertuples(index=False):
        comparable_lines.append(
            f"- ID {row.id}: prix {format_currency(row.price)}, "
            f"{format_number(row.bedrooms, 0)} chambres, "
            f"{format_number(row.bathrooms, 2)} sdb, "
            f"{format_number(row.sqft_living, 0)} pi2, zipcode {row.zipcode}"
        )

    return f"""
Tu es un analyste immobilier specialise en evaluation residentielle.
Evalue si la propriete suivante semble etre une opportunite d'investissement.

Propriete cible:
- ID: {selected_property['id']}
- Prix: {format_currency(selected_property['price'])}
- Chambres: {format_number(selected_property['bedrooms'], 0)}
- Salles de bains: {format_number(selected_property['bathrooms'], 2)}
- Superficie habitable: {format_number(selected_property['sqft_living'], 0)} pi2
- Prix au pi2: {format_currency(selected_property['price_per_sqft'])}
- Zipcode: {selected_property['zipcode']}
- Grade: {format_number(selected_property['grade'], 0)}
- Condition: {format_number(selected_property['condition'], 0)}
- Annee de construction: {format_number(selected_property['yr_built'], 0)}
- Renovee: {"Oui" if selected_property['is_renovated'] else "Non"}
- Sous-sol: {"Oui" if selected_property['has_basement'] else "Non"}
- Front de mer: {selected_property['waterfront_label']}

Analyse comparables:
- Regle appliquee: {rule_label}
- Nombre de comparables: {len(comparables)}
- Prix moyen comparables: {format_currency(avg_comp_price)}
- Ecart de prix: {format_currency_delta(gap_value)}
- Ecart relatif: {format_percent(gap_pct)}
- Verdict quantitatif: {classify_valuation(gap_pct)}

Exemples de comparables:
{os.linesep.join(comparable_lines)}

Instruction:
- Reponds en francais.
- Organise la reponse en 3 parties: forces, risques, recommandation finale.
- Termine par une decision claire: acheter, negocier ou eviter.
""".strip()


def generate_llm_response(prompt: str) -> str:
    api_key = get_api_key()
    if not api_key:
        return (
            "Configuration manquante: ajoutez GOOGLE_API_KEY dans .env ou dans "
            "les secrets Streamlit pour activer Gemini."
        )
    if genai is None:
        return (
            "La librairie google-genai n'est pas installee. Verifiez "
            "requirements.txt et relancez l'application."
        )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL),
            contents=prompt,
        )
        text = getattr(response, "text", "") or ""
        return text.strip() or "Le modele n'a pas retourne de texte exploitable."
    except Exception as exc:
        return f"Appel Gemini impossible pour le moment: {exc}"


def build_property_options(df: pd.DataFrame) -> dict[str, str]:
    options = {}
    for row in df.itertuples(index=False):
        options[row.id] = (
            f"ID {row.id} | {format_currency(row.price)} | "
            f"{format_number(row.bedrooms, 0)} ch | "
            f"{format_number(row.sqft_living, 0)} pi2 | {row.zipcode}"
        )
    return options


def find_comparables(
    market_df: pd.DataFrame, selected_property: pd.Series
) -> tuple[pd.DataFrame, str]:
    candidates = market_df[market_df["id"] != selected_property["id"]].copy()
    if candidates.empty:
        return candidates, "Aucun comparable disponible"

    target_sqft = max(float(selected_property["sqft_living"]), 1.0)
    target_bedrooms = float(selected_property["bedrooms"])
    target_bathrooms = float(selected_property["bathrooms"])
    target_ppsf = max(float(selected_property["price_per_sqft"]), 1.0)

    def score_frame(frame: pd.DataFrame) -> pd.DataFrame:
        scored = frame.copy()
        scored["comparison_score"] = (
            (scored["sqft_living"] - target_sqft).abs() / target_sqft
            + (scored["bedrooms"] - target_bedrooms).abs() * 0.45
            + (scored["bathrooms"] - target_bathrooms).abs() * 0.25
            + (scored["price_per_sqft"] - target_ppsf).abs() / target_ppsf
        )
        return scored.sort_values(["comparison_score", "date"])

    strategies = [
        (
            "Comparables stricts: meme zipcode, memes chambres, superficie +/- 20%",
            candidates[
                candidates["zipcode"].eq(selected_property["zipcode"])
                & candidates["bedrooms"].eq(target_bedrooms)
                & candidates["sqft_living"].between(target_sqft * 0.8, target_sqft * 1.2)
            ],
        ),
        (
            "Comparables elargis: meme zipcode, chambres +/- 1, superficie +/- 25%",
            candidates[
                candidates["zipcode"].eq(selected_property["zipcode"])
                & candidates["bedrooms"].between(
                    target_bedrooms - 1, target_bedrooms + 1
                )
                & candidates["sqft_living"].between(target_sqft * 0.75, target_sqft * 1.25)
            ],
        ),
        (
            "Comparables de secours: meme zipcode, chambres +/- 1, superficie +/- 35%",
            candidates[
                candidates["zipcode"].eq(selected_property["zipcode"])
                & candidates["bedrooms"].between(
                    target_bedrooms - 1, target_bedrooms + 1
                )
                & candidates["sqft_living"].between(target_sqft * 0.65, target_sqft * 1.35)
            ],
        ),
        (
            "Comparables larges: marche filtre, chambres +/- 1, superficie +/- 35%",
            candidates[
                candidates["bedrooms"].between(target_bedrooms - 1, target_bedrooms + 1)
                & candidates["sqft_living"].between(target_sqft * 0.65, target_sqft * 1.35)
            ],
        ),
    ]

    best_frame = candidates.head(0)
    best_label = "Aucun comparable trouve"

    for label, frame in strategies:
        if frame.empty:
            continue
        scored = score_frame(frame)
        if len(scored) >= 5:
            return scored.head(10).reset_index(drop=True), label
        if len(scored) > len(best_frame):
            best_frame = scored
            best_label = label

    return best_frame.head(10).reset_index(drop=True), best_label


def plot_property_vs_comparables(
    selected_property: pd.Series, comparables: pd.DataFrame
) -> plt.Figure:
    chart_df = comparables.head(6).copy()
    labels = ["Selectionnee"] + [
        f"Comparable {index + 1}" for index in range(len(chart_df))
    ]
    values = [selected_property["price"]] + chart_df["price"].tolist()
    colors = ["#d94841"] + ["#94a3b8"] * len(chart_df)

    fig_height = max(4.5, 0.7 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    bars = ax.barh(labels, values, color=colors)
    ax.axvline(
        chart_df["price"].mean(),
        color="#1f77b4",
        linestyle="--",
        linewidth=2,
        label="Moyenne comparables",
    )
    ax.set_title("Prix de la propriete vs comparables")
    ax.set_xlabel("Prix (USD)")
    ax.xaxis.set_major_formatter(FuncFormatter(dollar_axis_formatter))
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)

    for bar, value in zip(bars, values):
        ax.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f" {format_currency(value)}",
            va="center",
            fontsize=9,
        )

    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def render_market_tab(filtered_df: pd.DataFrame) -> None:
    st.subheader("Onglet 1 - Exploration du marche")
    if filtered_df.empty:
        st.warning("Aucune propriete ne correspond aux filtres actuels.")
        return

    st.caption(
        "Les indicateurs et graphiques ci-dessous se mettent a jour selon les "
        "filtres choisis dans la barre laterale."
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Nombre de proprietes", format_number(len(filtered_df), 0))
    metric_columns[1].metric("Prix moyen", format_currency(filtered_df["price"].mean()))
    metric_columns[2].metric("Prix median", format_currency(filtered_df["price"].median()))
    metric_columns[3].metric(
        "Prix moyen au pi2",
        format_currency(filtered_df["price_per_sqft"].mean()),
    )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        show_figure(plot_price_distribution(filtered_df))
    with chart_right:
        show_figure(plot_surface_vs_price(filtered_df))

    show_figure(plot_avg_price_by_bedrooms(filtered_df))

    st.markdown("### Resume genere par LLM")
    market_key = dataset_signature(filtered_df)
    if st.button("Generer un resume du marche", type="primary", key="market_summary"):
        with st.spinner("Analyse Gemini en cours..."):
            st.session_state["market_summary_payload"] = {
                "signature": market_key,
                "text": generate_llm_response(build_market_prompt(filtered_df)),
            }

    payload = st.session_state.get("market_summary_payload")
    if payload and payload.get("signature") == market_key:
        st.info(payload["text"])


def render_property_header(selected_property: pd.Series) -> None:
    sale_date = (
        selected_property["date"].strftime("%Y-%m-%d")
        if pd.notna(selected_property["date"])
        else "Inconnue"
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Prix", format_currency(selected_property["price"]))
    metric_columns[1].metric(
        "Superficie",
        f"{format_number(selected_property['sqft_living'], 0)} pi2",
    )
    metric_columns[2].metric(
        "Prix au pi2",
        format_currency(selected_property["price_per_sqft"]),
    )
    metric_columns[3].metric(
        "Annee de construction",
        format_number(selected_property["yr_built"], 0),
    )

    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown("**Caracteristiques principales**")
        st.write(f"- Chambres: {format_number(selected_property['bedrooms'], 0)}")
        st.write(f"- Salles de bains: {format_number(selected_property['bathrooms'], 2)}")
        st.write(f"- Etages: {format_number(selected_property['floors'], 2)}")
        st.write(f"- Zipcode: {selected_property['zipcode']}")
        st.write(f"- Date de vente: {sale_date}")

    with right_column:
        st.markdown("**Qualite et structure**")
        st.write(f"- Grade: {format_number(selected_property['grade'], 0)}")
        st.write(f"- Condition: {format_number(selected_property['condition'], 0)}")
        st.write(f"- Age au moment de la vente: {format_number(selected_property['age'], 0)} ans")
        st.write(f"- Renovee: {'Oui' if selected_property['is_renovated'] else 'Non'}")
        st.write(f"- Sous-sol: {'Oui' if selected_property['has_basement'] else 'Non'}")


def render_property_tab(filtered_df: pd.DataFrame) -> None:
    st.subheader("Onglet 2 - Analyse d'une propriete")
    if filtered_df.empty:
        st.warning("Elargissez les filtres pour selectionner une propriete.")
        return

    options = build_property_options(filtered_df)
    selected_id = st.selectbox(
        "Choisir une propriete a analyser",
        options=list(options.keys()),
        format_func=lambda property_id: options[property_id],
    )
    selected_property = filtered_df.loc[filtered_df["id"] == selected_id].iloc[0]

    render_property_header(selected_property)

    comparables, rule_label = find_comparables(filtered_df, selected_property)

    st.markdown("### Comparables")
    st.caption(f"Regle appliquee: {rule_label}")

    if comparables.empty:
        st.warning(
            "Aucun comparable n'a ete trouve avec les criteres actuels. "
            "Essayez d'elargir les filtres du marche."
        )
        return

    avg_comp_price = comparables["price"].mean()
    gap_value = selected_property["price"] - avg_comp_price
    gap_pct = gap_value / avg_comp_price if avg_comp_price else np.nan
    valuation_label = classify_valuation(gap_pct)

    metric_columns = st.columns(3)
    metric_columns[0].metric("Prix moyen comparables", format_currency(avg_comp_price))
    metric_columns[1].metric(
        "Ecart vs marche local",
        format_currency_delta(gap_value),
        format_percent(gap_pct),
    )
    metric_columns[2].metric("Verdict", valuation_label)

    comparable_table = comparables[
        [
            "id",
            "date",
            "price",
            "bedrooms",
            "bathrooms",
            "sqft_living",
            "price_per_sqft",
            "zipcode",
        ]
    ].copy()
    comparable_table["date"] = comparable_table["date"].dt.strftime("%Y-%m-%d")
    comparable_table["price"] = comparable_table["price"].map(format_currency)
    comparable_table["bedrooms"] = comparable_table["bedrooms"].map(
        lambda value: format_number(value, 0)
    )
    comparable_table["bathrooms"] = comparable_table["bathrooms"].map(
        lambda value: format_number(value, 2)
    )
    comparable_table["sqft_living"] = comparable_table["sqft_living"].map(
        lambda value: format_number(value, 0)
    )
    comparable_table["price_per_sqft"] = comparable_table["price_per_sqft"].map(
        format_currency
    )
    comparable_table.columns = [
        "ID",
        "Date",
        "Prix",
        "Chambres",
        "Sdb",
        "Superficie (pi2)",
        "Prix au pi2",
        "Zipcode",
    ]
    st.dataframe(comparable_table, use_container_width=True, hide_index=True)

    show_figure(plot_property_vs_comparables(selected_property, comparables))

    st.markdown("### Recommandation generee par LLM")
    current_signature = property_signature(selected_property, comparables)
    if st.button(
        "Generer une recommandation d'investissement",
        type="primary",
        key="property_recommendation",
    ):
        with st.spinner("Analyse Gemini en cours..."):
            st.session_state["property_recommendation_payload"] = {
                "signature": current_signature,
                "text": generate_llm_response(
                    build_property_prompt(selected_property, comparables, rule_label)
                ),
            }

    payload = st.session_state.get("property_recommendation_payload")
    if payload and payload.get("signature") == current_signature:
        st.success(payload["text"])


def main() -> None:
    st.set_page_config(
        page_title="King County Real Estate Analyzer",
        layout="wide",
    )

    st.title("King County Real Estate Analyzer")
    st.write(
        "Application Streamlit pour explorer le marche immobilier du comte de "
        "King et analyser des proprietes individuelles."
    )

    if not DATA_PATH.exists():
        st.error(
            "Le fichier kc_house_data.csv est introuvable dans le dossier du projet."
        )
        st.stop()

    df = load_data(str(DATA_PATH))
    filters = render_sidebar_filters(df)
    filtered_df = apply_filters(df, filters)

    st.caption(
        f"Jeu de donnees charge: {format_number(len(df), 0)} transactions | "
        f"Resultats filtres: {format_number(len(filtered_df), 0)}"
    )

    market_tab, property_tab = st.tabs(
        ["Onglet 1 - Marche", "Onglet 2 - Propriete"]
    )

    with market_tab:
        render_market_tab(filtered_df)

    with property_tab:
        render_property_tab(filtered_df)


if __name__ == "__main__":
    main()
