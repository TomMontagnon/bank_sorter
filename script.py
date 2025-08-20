#!/usr/bin/python3
from pathlib import Path
import pandas as pd
import re

SUFFIX = "_cat"
DATA_REP = "bank_data/"
RAW_DATA_FILENAME = "raw_data.txt"
PARSED_DATA_FILENAME = "parsed_data.csv"
CATEG_FILE_NAME = "categories.csv"


def init_overview() -> dict:
    overview = {
        "Janvier": {
            "has_raw_data": False,
            "is_parsed": False,
            "nb_categorized": 0,
            "nb_items": 0,
        },
        "Fevrier": {
            "has_raw_data": False,
            "is_parsed": False,
            "nb_categorized": 0,
            "nb_items": 0,
        },
        "Mars": {
            "has_raw_data": False,
            "is_parsed": False,
            "nb_categorized": 0,
            "nb_items": 0,
        },
        "Avril": {
            "has_raw_data": False,
            "is_parsed": False,
            "nb_categorized": 0,
            "nb_items": 0,
        },
        "Mai": {
            "has_raw_data": False,
            "is_parsed": False,
            "nb_categorized": 0,
            "nb_items": 0,
        },
        "Juin": {
            "has_raw_data": False,
            "is_parsed": False,
            "nb_categorized": 0,
            "nb_items": 0,
        },
        "Juillet": {
            "has_raw_data": False,
            "is_parsed": False,
            "nb_categorized": 0,
            "nb_items": 0,
        },
        "Aout": {
            "has_raw_data": False,
            "is_parsed": False,
            "nb_categorized": 0,
            "nb_items": 0,
        },
        "Septembre": {
            "has_raw_data": False,
            "is_parsed": False,
            "nb_categorized": 0,
            "nb_items": 0,
        },
        "Octobre": {
            "has_raw_data": False,
            "is_parsed": False,
            "nb_categorized": 0,
            "nb_items": 0,
        },
        "Novembre": {
            "has_raw_data": False,
            "is_parsed": False,
            "nb_categorized": 0,
            "nb_items": 0,
        },
        "Decembre": {
            "has_raw_data": False,
            "is_parsed": False,
            "nb_categorized": 0,
            "nb_items": 0,
        },
    }
    for key in list(overview.keys()):
        folder = Path(DATA_REP) / key
        folder.mkdir(parents=True, exist_ok=True)
        file = folder / RAW_DATA_FILENAME
        if file.is_file():
            overview[key]["has_raw_data"] = True
        file2 = folder / PARSED_DATA_FILENAME
        if file2.is_file():
            overview[key]["is_parsed"] = True

            data_frame = csv_to_df(file2.resolve())

            overview[key]["nb_items"] = len(data_frame)
            overview[key]["nb_categorized"] = (
                data_frame["CATEGORIE"].replace({None: pd.NA}).notna().sum()
            )
    return overview


def print_status(overview: dict, progress_col_name: str = "progress") -> None:
    df_overview = (
        pd.DataFrame.from_dict(overview, orient="index")
        .reset_index()
        .rename(columns={"index": "Mois"})
    )

    # Colonnes bool à convertir (détection + fallback sur noms attendus)
    bool_cols = [
        c for c in df_overview.columns if pd.api.types.is_bool_dtype(df_overview[c])
    ]
    for c in ["has_raw_data", "is_parsed"]:
        if c in df_overview.columns and c not in bool_cols:
            bool_cols.append(c)

    # Mapping bool -> emojis
    emoji_map = {True: "✅     ", False: "🛑     "}
    for c in bool_cols:
        df_overview[c] = df_overview[c].astype("boolean").map(emoji_map)

    # Colonne de progression "x / y"
    if {"nb_categorized", "nb_items"}.issubset(df_overview.columns):
        df_overview[progress_col_name] = (
            df_overview["nb_categorized"].fillna(0).astype(int).astype(str)
            + " / "
            + df_overview["nb_items"].fillna(0).astype(int).astype(str)
            + "   "
        )
        df_overview = df_overview.drop(columns=["nb_categorized", "nb_items"])

    # Ordre des colonnes: Mois, bools, progress, puis le reste
    ordered = []
    if "Mois" in df_overview.columns:
        ordered.append("Mois")
    ordered += [c for c in bool_cols if c in df_overview.columns and c != "Mois"]
    if progress_col_name in df_overview.columns:
        ordered.append(progress_col_name)
    ordered += [c for c in df_overview.columns if c not in ordered]

    print(df_overview[ordered].to_string(index=True))


def my_menu(overview: dict) -> int:
    print_status(overview)

    while True:
        choix = input("Entrez le numéro (0-11) ou 'q' pour quitter : ")
        if choix.lower() == "q":
            print("Abandon.")
            return None
        if choix.isdigit():
            n = int(choix)
            if 0 <= n <= 11:
                val = list(overview.keys())[n]
                print("Vous avez choisi :", val)
                return val
        print("Choix invalide, réessayez.")


def read_categ_csv(item_sep: str = "|") -> dict[str, list[str]]:
    """
    Lit le CSV à deux colonnes 'clé' et 'items' et reconstruit le dict.
    """
    path = Path(DATA_REP) / CATEG_FILE_NAME
    df = pd.read_csv(path.resolve(), sep=";", encoding="utf-8")
    # Remplace NaN ou '' par liste vide, sinon split
    df["items"] = df["items"].fillna("").apply(lambda s: s.split(item_sep) if s else [])
    return df


def write_categ_csv(
    df: pd.DataFrame,
    key_col: str = "clef",
    items_col: str = "items",
    item_sep: str = "|",
) -> None:
    df2 = df.copy()

    def _join_items(x):
        if isinstance(x, (list, tuple)):
            return item_sep.join(x)
        if pd.isna(x):
            return ""
        return str(x)

    df2[items_col] = df2[items_col].apply(_join_items)
    df2 = df2[[key_col, items_col]]
    path = Path(DATA_REP) / CATEG_FILE_NAME
    df2.to_csv(path.resolve(), index=False, sep=";", encoding="utf-8")


def auto_catego(data_key: str) -> None:
    path = Path(DATA_REP) / data_key / PARSED_DATA_FILENAME
    data_frame = csv_to_df(path.resolve())
    data_frame["MONTANT"] = (
        data_frame["PRIX_STR"]
        .str.replace("EUR", "")
        .str.replace(",", ".")
        .astype(float)
    )
    categ = read_categ_csv()
    amount_rules = []

    def catego(desc: str) -> str:
        desc = desc.lower()
        return next(
            (
                clef
                for clef, items in zip(categ["clef"], categ["items"])
                if any(kw.lower() in desc for kw in items)
            ),
            "",  # valeur par défaut si aucun match
        )

    def amount_catego(row):
        desc = row["DESCRIPTION"].lower()
        montant = f"{row['MONTANT']:.2f}"
        for keyword, category in amount_rules:
            if keyword in desc or keyword in montant:
                return category
        return catego(desc)

    data_frame["CATEGORIE"] = data_frame.apply(amount_catego, axis=1)

    df_to_csv(
        data_frame,
        Path(DATA_REP) / data_key / PARSED_DATA_FILENAME,
    )


def df_to_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(
        name,  # nom du fichier de sortie
        sep=";",  # séparateur (',' par défaut)
        index=False,  # ne pas écrire la colonne d'index
        encoding="utf-8",  # encodage du fichier
    )
    print(f"{name} créé !")


def csv_to_df(name: str) -> pd.DataFrame:
    return pd.read_csv(
        name,
        sep=";",  # séparateur (',' par défaut)
        encoding="utf-8",  # encodage du fichier
    )


def display_bilan(data_key: str) -> None:
    path = Path(DATA_REP) / data_key / PARSED_DATA_FILENAME
    data_frame = csv_to_df(path.resolve())
    somme = data_frame.groupby("CATEGORIE")["MONTANT"].sum()
    print(somme)
    # Somme des valeurs positives (revenus/remboursements)
    somme_positives = data_frame[data_frame["MONTANT"] > 0]["MONTANT"].sum()

    # Somme des valeurs négatives (dépenses)
    somme_negatives = data_frame[data_frame["MONTANT"] < 0]["MONTANT"].sum()

    print("\nSomme des valeurs positives :", somme_positives)
    print("Somme des valeurs négatives :", somme_negatives)


def parse_data(data_key: str) -> None:
    # Load the file content
    file_path = Path(DATA_REP) / data_key / RAW_DATA_FILENAME
    with Path.open(file_path, encoding="utf-8") as f:
        lines = f.readlines()

    # Regex patterns
    date_pattern = re.compile(r"^\d{2}/\d{2}/\d{4}$")
    amount_pattern = re.compile(r"^[\t ]*[+-]?\d{1,3}(?:[\s\d]*,\d{2}) EUR[\t ]*$")

    # Parse entries
    entries = []
    current_date = None
    current_lines = []

    for line in lines:
        sline = line.strip()
        if date_pattern.match(sline):
            current_date = sline
            current_lines = []
        elif amount_pattern.match(sline):
            description = " ".join(current_lines).strip()
            price = sline.replace("\t", "").replace(" ", "")
            entries.append((current_date, description, price, ""))
            current_lines = []
        else:
            current_lines.append(sline)

    df_to_csv(
        pd.DataFrame(entries, columns=["DATE", "DESCRIPTION", "PRIX_STR", "CATEGORIE"]),
        Path(DATA_REP) / data_key / PARSED_DATA_FILENAME,
    )


def main() -> None:
    while True:
        overview = init_overview()
        data_key = my_menu(overview)
        if data_key is None:
            return
        if not overview[data_key]["has_raw_data"]:
            print(f"Has no raw data for : {data_key}")
            continue
        if not overview[data_key]["is_parsed"]:
            parse_data(data_key)
            continue
        if overview[data_key]["nb_categorized"] == 0:
            auto_catego(data_key)
        elif overview[data_key]["nb_categorized"] == overview[data_key]["nb_items"]:
            display_bilan(data_key)
        else:
            print("Veuillez finir de categoriser le items")


if __name__ == "__main__":
    main()
