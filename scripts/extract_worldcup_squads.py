import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import pdfplumber


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_utils import canonical_team_name


DEFAULT_PDF_PATH = ROOT / "data" / "squads_pdf" / "SquadLists-Spanish.pdf"
OUTPUT_PATH = ROOT / "data" / "worldcup_squads.csv"


PLAYER_COLUMNS = [
    "number",
    "position",
    "player_name",
    "first_names",
    "last_names",
    "shirt_name",
    "birth_date",
    "club",
    "height_cm",
]


def clean_cell(value):
    if value is None:
        return ""

    return " ".join(str(value).replace("\x00", "").split()).strip()


def compact_row(row):
    return [clean_cell(cell) for cell in row if clean_cell(cell)]


def extract_team_name(page_text):
    if not page_text:
        return None, None

    for line in page_text.splitlines():
        line = clean_cell(line)
        match = re.fullmatch(r"(.+)\s+\(([A-Z]{3})\)", line)

        if match:
            raw_team = match.group(1)
            return canonical_team_name(raw_team), match.group(2)

    return None, None


def parse_player_row(cells):
    if len(cells) < 9 or not cells[0].isdigit():
        return None

    birth_date = pd.to_datetime(cells[6], dayfirst=True, errors="coerce")

    return {
        "number": int(cells[0]),
        "position": cells[1],
        "player_name": cells[2],
        "first_names": cells[3],
        "last_names": cells[4],
        "shirt_name": cells[5],
        "birth_date": birth_date.date().isoformat()
        if not pd.isna(birth_date)
        else cells[6],
        "club": cells[7],
        "height_cm": int(float(cells[8])) if str(cells[8]).isdigit() else None,
    }


def parse_coach_row(cells):
    if len(cells) < 2:
        return None

    if clean_cell(cells[0]).lower() != "entrenador":
        return None

    return {
        "coach_name": cells[1],
        "coach_first_names": cells[2] if len(cells) > 2 else "",
        "coach_last_names": cells[3] if len(cells) > 3 else "",
        "coach_nationality": cells[4] if len(cells) > 4 else "",
    }


def extract_squads(pdf_path):
    player_rows = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            team, team_code = extract_team_name(page_text)

            if not team:
                print(f"Warning: team not detected on page {page_number}")
                continue

            coach = {
                "coach_name": "",
                "coach_first_names": "",
                "coach_last_names": "",
                "coach_nationality": "",
            }
            tables = page.extract_tables()

            for table in tables:
                for row in table:
                    cells = compact_row(row)
                    player = parse_player_row(cells)

                    if player:
                        player_rows.append({
                            "team": team,
                            "team_code": team_code,
                            "page": page_number,
                            **player,
                            **coach,
                        })
                        continue

                    parsed_coach = parse_coach_row(cells)

                    if parsed_coach:
                        coach.update(parsed_coach)

            for row in player_rows:
                if row["team"] == team and not row["coach_name"]:
                    row.update(coach)

    return pd.DataFrame(player_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default=str(DEFAULT_PDF_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    output_path = Path(args.output)

    squads = extract_squads(pdf_path)

    if squads.empty:
        raise RuntimeError("No player rows were extracted from the PDF.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    squads.to_csv(output_path, index=False, encoding="utf-8")

    team_counts = squads.groupby("team")["player_name"].count().reset_index()
    incomplete = team_counts[team_counts["player_name"] != 26]

    print(f"Wrote {output_path.relative_to(ROOT)}")
    print(f"Players extracted: {len(squads)}")
    print(f"Teams extracted: {squads['team'].nunique()}")

    if incomplete.empty:
        print("All teams have 26 players.")
    else:
        print("Teams with player count different from 26:")
        print(incomplete.to_string(index=False))


if __name__ == "__main__":
    main()
