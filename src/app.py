from __future__ import annotations

import requests
import streamlit as st
from streamlit import components

from map_engine import load_fixed_graph, build_routes, build_map


@st.cache_data(show_spinner=False, ttl=3600)
def geocode_nominatim(query: str, limit: int = 8) -> list[tuple[str, float, float]]:
    q = query.strip()
    if not q:
        return []

    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": q, "format": "jsonv2", "limit": limit, "addressdetails": 0}
    headers = {"User-Agent": "route-planner-streamlit/1.0"}

    r = requests.get(url, params=params, headers=headers, timeout=15)
    r.raise_for_status()
    data = r.json()

    hits: list[tuple[str, float, float]] = []
    for item in data:
        label = item.get("display_name", "unknown")
        lat = float(item["lat"])
        lon = float(item["lon"])
        hits.append((label, lat, lon))
    return hits


def _search_callback(which: str) -> None:
    q = (st.session_state.get(f"{which}_query") or "").strip()
    st.session_state[f"{which}_error"] = ""

    if len(q) < 3:
        st.session_state[f"{which}_hits"] = []
        st.session_state[f"{which}_choice"] = None
        st.session_state[f"{which}_point"] = None
        return

    try:
        hits = geocode_nominatim(q, limit=10)
        st.session_state[f"{which}_hits"] = hits
        st.session_state[f"{which}_choice"] = 0 if hits else None
        st.session_state[f"{which}_point"] = (hits[0][1], hits[0][2]) if hits else None
    except Exception as e:
        st.session_state[f"{which}_hits"] = []
        st.session_state[f"{which}_choice"] = None
        st.session_state[f"{which}_point"] = None
        st.session_state[f"{which}_error"] = str(e)


def cached_graph(network_type: str):
    return load_fixed_graph(network_type)


st.set_page_config(page_title="Route Planner", layout="wide")
st.title("Построение маршрута")

for key, default in [
    ("start_query", ""),
    ("end_query", ""),
    ("start_hits", []),
    ("end_hits", []),
    ("start_choice", None),
    ("end_choice", None),
    ("start_point", None),
    ("end_point", None),
    ("start_error", ""),
    ("end_error", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

top1, top2, top3 = st.columns([0.22, 0.22, 0.56])
with top1:
    network_type = st.selectbox("Тип сети", ["walk", "bike", "drive"], index=0)
with top2:
    optimize = st.selectbox("Оптимизация", ["distance", "time"], index=0)

left, right = st.columns(2)

with left:
    st.subheader("Старт")
    st.text_input(
        "Поиск старта",
        key="start_query",
        placeholder="Например: Старый Петергоф, станция",
        label_visibility="collapsed",
        on_change=lambda: _search_callback("start"),
    )

    if st.session_state["start_error"]:
        st.error(f"Ошибка поиска: {st.session_state['start_error']}")

    start_hits: list[tuple[str, float, float]] = st.session_state["start_hits"]
    if start_hits:
        idx = st.selectbox(
            "Варианты старта",
            options=list(range(len(start_hits))),
            format_func=lambda i: start_hits[i][0],
            key="start_choice",
        )
        st.session_state["start_point"] = (start_hits[idx][1], start_hits[idx][2])
        st.caption(f"{st.session_state['start_point'][0]:.6f}, {st.session_state['start_point'][1]:.6f}")
    else:
        if st.session_state["start_query"].strip():
            st.caption("Нажми Enter (минимум 3 символа).")

with right:
    st.subheader("Финиш")
    st.text_input(
        "Поиск финиша",
        key="end_query",
        placeholder="Например: Старый Петергоф, Сергиевка",
        label_visibility="collapsed",
        on_change=lambda: _search_callback("end"),
    )

    if st.session_state["end_error"]:
        st.error(f"Ошибка поиска: {st.session_state['end_error']}")

    end_hits: list[tuple[str, float, float]] = st.session_state["end_hits"]
    if end_hits:
        idx = st.selectbox(
            "Варианты финиша",
            options=list(range(len(end_hits))),
            format_func=lambda i: end_hits[i][0],
            key="end_choice",
        )
        st.session_state["end_point"] = (end_hits[idx][1], end_hits[idx][2])
        st.caption(f"{st.session_state['end_point'][0]:.6f}, {st.session_state['end_point'][1]:.6f}")
    else:
        if st.session_state["end_query"].strip():
            st.caption("Нажми Enter (минимум 3 символа).")

st.divider()

start = st.session_state["start_point"]
end = st.session_state["end_point"]

if not start or not end:
    st.info("Введи старт и финиш и нажми Enter в каждой строке.")
    st.stop()

if start == end:
    st.warning("Старт и финиш совпадают.")
    m = build_map([start, end], [], ["Start", "End"], zoom_start=14)
    components.v1.html(m.get_root().render(), height=680, scrolling=False)
    st.stop()

weight = "travel_time" if optimize == "time" else "length"

latlons = [start, end]
names = ["Start", "End"]

try:
    with st.spinner("Загрузка графа и построение маршрута..."):
        G = cached_graph(network_type)
        routes, errors = build_routes(G, latlons, weight=weight)

    if errors:
        st.warning("Не получилось построить маршрут:\n" + "\n".join(errors))
        st.info("Попробуй увеличить радиус или сменить тип сети (walk/drive).")

    m = build_map(latlons, routes, names, zoom_start=14)
    components.v1.html(m.get_root().render(), height=680, scrolling=False)

except Exception as e:
    st.error(f"Ошибка при загрузке графа/построении маршрута:\n\n{e}")
