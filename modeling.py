# modeling.py
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["doy"] = d["observed_on"].dt.dayofyear
    d["month"] = d["observed_on"].dt.month
    # rough geographic transforms
    d["lat2"] = d["lat"]**2
    d["lon2"] = d["lon"]**2
    d["lat_lon"] = d["lat"] * d["lon"]
    return d

def sample_background(df: pd.DataFrame, n: int = None, seed: int = 42) -> pd.DataFrame:
    """Uniformly sample background points within the min/max bbox of presence data."""
    rng = np.random.default_rng(seed)
    lat_min, lat_max = df["lat"].min(), df["lat"].max()
    lon_min, lon_max = df["lon"].min(), df["lon"].max()
    if n is None:
        n = len(df)
    bg = pd.DataFrame({
        "lat": rng.uniform(lat_min, lat_max, n),
        "lon": rng.uniform(lon_min, lon_max, n),
        "observed_on": rng.choice(df["observed_on"].values, n)  # randomize time from presence
    })
    bg["label"] = 0
    return bg

def train_presence_model(presence_df: pd.DataFrame, seed: int = 42) -> Tuple[XGBClassifier, float]:
    pres = presence_df.copy()
    pres["label"] = 1
    bg = sample_background(pres, n=len(pres), seed=seed)
    data = pd.concat([pres[["lat","lon","observed_on"]], bg[["lat","lon","observed_on","label"]]], ignore_index=True)
    data["observed_on"] = pd.to_datetime(data["observed_on"], utc=True, errors="coerce")
    data = add_basic_features(data)
    data["label"] = data["label"].fillna(1).astype(int)
    X = data[["lat","lon","lat2","lon2","lat_lon","month","doy"]]
    y = data["label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    clf = XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=seed
    )
    clf.fit(Xtr, ytr)
    auc = roc_auc_score(yte, clf.predict_proba(Xte)[:,1])
    return clf, auc

def predict_grid(clf: XGBClassifier, bbox: Tuple[float,float,float,float], month: int, step: float = 0.5) -> pd.DataFrame:
    lat_min, lat_max, lon_min, lon_max = bbox
    lats = np.arange(lat_min, lat_max + step, step)
    lons = np.arange(lon_min, lon_max + step, step)
    grid = np.array([(la, lo) for la in lats for lo in lons])
    df = pd.DataFrame(grid, columns=["lat","lon"])
    df["month"] = month
    df["doy"] = int(30.5 * (month-1) + 15)  # mid-month approx
    df["lat2"] = df["lat"]**2
    df["lon2"] = df["lon"]**2
    df["lat_lon"] = df["lat"] * df["lon"]
    df["score"] = clf.predict_proba(df[["lat","lon","lat2","lon2","lat_lon","month","doy"]])[:,1]
    return df

def cluster_hotspots(df: pd.DataFrame, eps_km: float = 50, min_samples: int = 10) -> pd.DataFrame:
    """
    Cluster observations into hotspots with DBSCAN using haversine (approx).
    Convert eps_km to degrees latitude (~111km/deg) for a quick approximation.
    """
    deg = eps_km / 111.0
    coords = df[["lat","lon"]].to_numpy()
    # scale lon by cos(lat) to reduce distortion
    mean_lat_rad = np.deg2rad(df["lat"].mean())
    coords_scaled = coords.copy()
    coords_scaled[:,1] = coords_scaled[:,1] * np.cos(mean_lat_rad)
    labels = DBSCAN(eps=deg, min_samples=min_samples).fit_predict(coords_scaled)
    out = df.copy()
    out["cluster"] = labels
    return out
