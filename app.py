# app.py
import streamlit as st
import pandas as pd
import pydeck as pdk
from datetime import date, timedelta
from inat_api import search_taxon_ids, fetch_observations, COMMON_SPECIES
from modeling import train_presence_model, predict_grid, cluster_hotspots

st.set_page_config(page_title="Wildlife Insights (iNaturalist)", layout="wide")

st.title("Wildlife Insights — iNaturalist Explorer")
st.caption("Live observations → mapping, clustering, and quick habitat modeling")

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    default_species = ["Equus ferus przewalskii", "Equus caballus", "Carcharodon carcharias"]
    species_names = st.multiselect("Species (type to add):", options=COMMON_SPECIES, default=default_species)
    custom_add = st.text_input("Add species by name (comma-separated):", "")
    if custom_add.strip():
        species_names += [s.strip() for s in custom_add.split(",") if s.strip()]

    colA, colB = st.columns(2)
    with colA:
        d1 = st.date_input("Start date", value=date.today() - timedelta(days=120))
    with colB:
        d2 = st.date_input("End date", value=date.today())

    quality = st.selectbox("Quality grade", ["any","research","needs_id","casual"], index=0)
    per_species = st.slider("Max observations per species", 50, 1000, 400, step=50)

    st.divider()
    st.header("Modeling")
    do_cluster = st.checkbox("Hotspot clustering (DBSCAN)", True)
    do_model   = st.checkbox("Presence model (XGBoost)", False)
    month_for_grid = st.slider("Month for suitability grid", 1, 12, 7)

#st.info("Tip: start with 1–3 species and ~3–6 months for quicker results.")

# Resolve taxon IDs
if not species_names:
    st.warning("Pick at least one species.")
    st.stop()

with st.spinner("Resolving taxa..."):
    name_to_id = search_taxon_ids(species_names)
if not name_to_id:
    st.error("Could not resolve any species names.")
    st.stop()

# Fetch data
frames = []
for name, taxon_id in name_to_id.items():
    with st.spinner(f"Fetching observations: {name}"):
        df = fetch_observations(
            taxon_id=taxon_id,
            d1=d1.isoformat(),
            d2=d2.isoformat(),
            per_page=200,
            pages=max(1, per_species // 200),
            geo_only=True,
            quality=None if quality=="any" else quality
        )
        df["query_name"] = name
        frames.append(df)

df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
if df.empty:
    st.warning("No observations returned. Try expanding date range or removing filters.")
    st.stop()

st.success(f"Fetched {len(df):,} observations for {len(name_to_id)} species")

# Show table
with st.expander("Preview data", expanded=False):
    st.dataframe(df.head(100))

# Map
st.subheader("Map")
# Color per species
species_list = sorted(df["species"].dropna().unique().tolist())
palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b",
           "#e377c2","#7f7f7f","#bcbd22","#17becf"] * 3
color_map = {sp: list(int(palette[i % len(palette)][j+1:j+3],16) for j in (0,2,4)) for i, sp in enumerate(species_list)}

df_map = df.dropna(subset=["lat","lon"]).copy()
df_map["color"] = df_map["species"].map(color_map).apply(lambda rgb: [*rgb, 140])

view = pdk.ViewState(latitude=float(df_map["lat"].mean()),
                     longitude=float(df_map["lon"].mean()),
                     zoom=2)
layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_map,
    get_position='[lon, lat]',
    get_fill_color='color',
    get_radius=40000,  # meters
    pickable=True
)
tooltip = {"text": "{common_name}\n{species}\n{observed_on}\n{place_guess}"}
st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip))

# Clustering (optional)
if do_cluster:
    st.subheader("Hotspot clustering (DBSCAN)")
    eps_km = st.slider("Neighborhood (km)", 10, 150, 50, step=10)
    min_samples = st.slider("Min samples/cluster", 5, 50, 15, step=5)
    clustered = cluster_hotspots(df_map, eps_km=eps_km, min_samples=min_samples)
    st.write("Clusters (label -1 = noise):")
    st.dataframe(clustered["cluster"].value_counts())
    # quick cluster layer
    clrs = {c: [int(50+205*(i%2==0)), int(50+205*(i%3==0)), int(50+205*(i%5==0)), 180]
            for i, c in enumerate(sorted(clustered["cluster"].unique()))}
    clustered["cl_color"] = clustered["cluster"].map(clrs)
    cl_layer = pdk.Layer("ScatterplotLayer", data=clustered,
                         get_position='[lon, lat]', get_fill_color='cl_color', get_radius=50000, pickable=True)
    st.pydeck_chart(pdk.Deck(layers=[cl_layer], initial_view_state=view, tooltip=tooltip))

# Presence model (optional)
if do_model:
    st.subheader("Presence-background habitat model")
    # train per species (first species only to keep it quick)
    target_species = st.selectbox("Train for species", species_list, index=0)
    pres = df_map[df_map["species"] == target_species].copy()
    if len(pres) < 60:
        st.warning("Need ≥ 60 presences for a minimally useful model.")
    else:
        from modeling import train_presence_model, predict_grid
        clf, auc = train_presence_model(pres)
        st.write(f"Model AUC: **{auc:.3f}** (presence vs background)")
        # grid over bbox
        bbox = (float(pres["lat"].min()), float(pres["lat"].max()),
                float(pres["lon"].min()), float(pres["lon"].max()))
        grid = predict_grid(clf, bbox=bbox, month=month_for_grid, step=1.0)
        # visualize as heat
        grid["rgba"] = grid["score"].apply(lambda s: [255, int(255*(1-s)), 0, int(50+205*s)])
        heat_layer = pdk.Layer("ScatterplotLayer", data=grid,
                               get_position='[lon, lat]', get_fill_color='rgba',
                               get_radius=60000, pickable=False)
        st.pydeck_chart(pdk.Deck(layers=[heat_layer], initial_view_state=view))

# Downloads
st.subheader("Download")
st.download_button("Download observations (CSV)", df.to_csv(index=False).encode(), file_name="inat_observations.csv", mime="text/csv")
