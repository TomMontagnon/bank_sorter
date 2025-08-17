#!/usr/bin/python3
from pathlib import Path
import pandas as pd
import re

SUFFIX = "_cat"
CATEG_FILE_NAME = "categories.csv"
DATA_REP = "bank_data/"
CATEG_FILE_NAME = "categories.csv"

def lister_et_choisir_txt() -> Path:
    # 1) convertir en Path
    p = Path(DATA_REP)

    # 2) lister tous les .txt du dossier (non-récursif)
    fichiers = sorted([f for f in p.iterdir() if f.is_file()])

    if not fichiers:
        print(f"Aucun fichier .txt trouvé dans {p.resolve()}")
        return None

    # 3) afficher la liste numérotée
    print(f"Fichiers .txt disponibles dans {p.resolve()}:")
    for idx, fichier in enumerate(fichiers, start=1):
        print(f"  {idx:2d}. {fichier.name}")

    # 4) menu de sélection
    while True:
        choix = input(f"Entrez le numéro (1-{len(fichiers)}) ou 'q' pour quitter : ")
        if choix.lower() == "q":
            print("Abandon.")
            return None
        if choix.isdigit():
            n = int(choix)
            if 1 <= n <= len(fichiers):
                selection = fichiers[n - 1]
                print("Vous avez choisi :", selection.name)
                return selection
        print("Choix invalide, réessayez.")


def read_categ_csv(
    path: str = CATEG_FILE_NAME, sep: str = ";", item_sep: str = "|"
) -> dict[str, list[str]]:
    """
    Lit le CSV à deux colonnes 'clé' et 'items' et reconstruit le dict.
    """
    df = pd.read_csv(path, sep=sep, encoding="utf-8")
    # Remplace NaN ou '' par liste vide, sinon split
    df["items"] = df["items"].fillna("").apply(lambda s: s.split(item_sep) if s else [])
    return df


def write_categ_csv(
    df: pd.DataFrame,
    path: str = CATEG_FILE_NAME,
    key_col: str = "clef",
    items_col: str = "items",
    sep: str = ";",
    item_sep: str = "|",
    encoding: str | None = "utf-8",
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
    df2.to_csv(path, index=False, sep=sep, encoding=encoding)


def txt_to_df(name: str) -> pd.DataFrame:
    # Load the file content
    file_path = Path(name)
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
        l = line.strip()
        if date_pattern.match(l):
            current_date = l
            current_lines = []
        elif amount_pattern.match(l):
            description = " ".join(current_lines).strip()
            price = l.replace("\t", "").replace(" ", "")
            entries.append((current_date, description, price))
            current_lines = []
        else:
            current_lines.append(l)

    # Create DataFrame
    df = pd.DataFrame(entries, columns=["DATE", "DESCRIPTION", "PRIX_STR"])

    return df


def add_catego(df):
    df["MONTANT"] = (
        df["PRIX_STR"].str.replace("EUR", "").str.replace(",", ".").astype(float)
    )
    categ = read_categ_csv()
    amount_rules = []

    def catego(desc: str) -> str:
        desc = desc.lower()
        return next(
            (
                clé
                for clé, items in zip(categ["clef"], categ["items"])
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

    df["CATEGORIE"] = df.apply(amount_catego, axis=1)
    return df


def df_to_csv(df, name):
    df.to_csv(
        name,  # nom du fichier de sortie
        sep=";",  # séparateur (',' par défaut)
        index=False,  # ne pas écrire la colonne d’index
        encoding="utf-8",  # encodage du fichier
    )
    print(f"{name} créé !")


def csv_to_df(name):
    df = pd.read_csv(
        name,
        sep=";",  # séparateur (',' par défaut)
        encoding="utf-8",  # encodage du fichier
    )
    return df


def main():
    # on peut passer le dossier en argument, sinon on prend le dossier courant
    while True:
        fic = lister_et_choisir_txt()
        sfic = str(fic)
        if sfic.endswith(".txt"):
            df = txt_to_df(fic)
            df_to_csv(df, fic.parent / f"{fic.stem}.csv")
        elif sfic.endswith(".csv"):
            if not sfic.endswith(f"{SUFFIX}.csv"):
                df = csv_to_df(fic)
                df = add_catego(df)
                df_to_csv(df, fic.parent / f"{fic.stem}{SUFFIX}.csv")
            else:
                df = csv_to_df(fic)
                somme = df.groupby("CATEGORIE")["MONTANT"].sum()
                print(somme)
                # Somme des valeurs positives (revenus/remboursements)
                somme_positives = df[df["MONTANT"] > 0]["MONTANT"].sum()

                # Somme des valeurs négatives (dépenses)
                somme_negatives = df[df["MONTANT"] < 0]["MONTANT"].sum()

                print("\nSomme des valeurs positives :", somme_positives)
                print("Somme des valeurs négatives :", somme_negatives)
        else:
            print("extension non-reconnu")


if __name__ == "__main__":
    main()
