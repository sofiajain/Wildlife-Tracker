# inat_api.py
import requests
import pandas as pd
from typing import List, Dict, Optional

BASE = "https://api.inaturalist.org/v1"

# --- preset species list for dropdowns / demos ---
COMMON_SPECIES = [
    # Horses
    "Equus ferus przewalskii", "Equus ferus", "Equus caballus", "Equus kiang",
    # Marine mammals
    "Tursiops truncatus", "Megaptera novaeangliae", "Delphinus delphis",
    # Sharks & turtles
    "Carcharodon carcharias", "Rhincodon typus", "Caretta caretta", "Dermochelys coriacea",
    # Iconic birds (good for demos)
    "Haliaeetus leucocephalus", "Sterna paradisaea"
]

def search_taxon_ids(names: List[str]) -> Dict[str, int]:
    """Resolve species/common names → iNat taxon_id using /taxa."""
    out = {}
    for n in names:
        r = requests.get(f"{BASE}/taxa", params={"q": n, "per_page": 1, "rank": "species"})
        r.raise_for_status()
        res = r.json().get("results", [])
        if res:
            out[n] = res[0]["id"]
    return out

def fetch_observations(
    taxon_id: int,
    d1: str, d2: str,
    per_page: int = 200,   # 200 is iNat max
    pages: int = 5,        # pull up to 1000 rows/species by default
    geo_only: bool = True,
    quality: Optional[str] = None  # "research" | "needs_id" | "casual"
) -> pd.DataFrame:
    params = {
        "taxon_id": taxon_id,
        "per_page": per_page,
        "order_by": "created_at",
        "order": "desc",
        "d1": d1,
        "d2": d2,
    }
    if geo_only:
        params["geo"] = "true"
    if quality:
        params["quality_grade"] = quality

    rows = []
    for page in range(1, pages + 1):
        params["page"] = page
        r = requests.get(f"{BASE}/observations", params=params, timeout=60)
        r.raise_for_status()
        data = r.json().get("results", [])
        if not data:
            break
        for obs in data:
            if not obs.get("geojson"):
                continue
            rows.append({
                "species": obs["taxon"]["name"] if obs.get("taxon") else None,
                "common_name": obs.get("taxon", {}).get("preferred_common_name"),
                "observed_on": obs.get("observed_on"),
                "lat": obs["geojson"]["coordinates"][1],
                "lon": obs["geojson"]["coordinates"][0],
                "place_guess": obs.get("place_guess"),
                "user": obs.get("user", {}).get("login"),
                "quality_grade": obs.get("quality_grade"),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["observed_on"] = pd.to_datetime(df["observed_on"], errors="coerce", utc=True)
        df = df.dropna(subset=["lat","lon","observed_on"]).reset_index(drop=True)
    return df
