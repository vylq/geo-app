from __future__ import annotations

from collections import Counter

import requests
import streamlit as st
from streamlit import components
from sqlalchemy import bindparam, text

from map_engine import build_map, build_routes

try:
    from postgis_engine import get_postgis_engine_or_none, graph_in_db, route_between_db
except Exception:
    get_postgis_engine_or_none = None
    graph_in_db = None
    route_between_db = None


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


def _format_distance_m(meters: float | None) -> str:
    if meters is None:
        return ""
    if meters >= 1000:
        return f"{meters / 1000:.2f} км"
    return f"{meters:.0f} м"


def _format_duration_s(seconds: float | None) -> str:
    if seconds is None:
        return ""
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h} ч {m} мин"
    if m > 0:
        return f"{m} мин"
    return f"{sec} с"


def _route_totals_from_edges_table(engine, edges_table: str, edge_ids: list[int]) -> tuple[float, float] | None:
    if not engine or not edge_ids:
        return None

    counts = Counter(int(x) for x in edge_ids)
    uniq = list(counts.keys())

    stmt = (
        text(f"SELECT id, cost_len, cost_time FROM {edges_table} WHERE id IN :ids")
        .bindparams(bindparam("ids", expanding=True))
    )

    try:
        with engine.begin() as conn:
            rows = conn.execute(stmt, {"ids": uniq}).mappings().all()
    except Exception:
        return None

    dist_m = 0.0
    time_s = 0.0
    for r in rows:
        c = counts.get(int(r["id"]), 0)
        dist_m += float(r["cost_len"]) * c
        time_s += float(r["cost_time"]) * c

    return dist_m, time_s


st.set_page_config(page_title="Route Planner", layout="wide")

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

left_col, right_col = st.columns([0.38, 0.62], gap="large")

with left_col:
    st.title("Построение маршрута")

    c1, c2 = st.columns(2)
    with c1:
        network_type = st.selectbox("Тип сети", ["walk", "bike", "drive"], index=0)
    with c2:
        optimize = st.selectbox("Оптимизация", ["distance", "time"], index=0)

    st.divider()

    s_col, e_col = st.columns(2)

    with s_col:
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

    with e_col:
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

start = st.session_state["start_point"]
end = st.session_state["end_point"]

weight = "travel_time" if optimize == "time" else "length"
latlons = [start, end] if start and end else []
names = ["Start", "End"]

routes = []
errors: list[str] = []
db_used = False
route_totals: tuple[float, float] | None = None

engine = None
if get_postgis_engine_or_none is not None:
    try:
        engine = get_postgis_engine_or_none()
    except Exception:
        engine = None

db_available = bool(engine) and (graph_in_db is not None) and (route_between_db is not None)

if start and end and start != end:
    if db_available and graph_in_db(engine, network_type):
        db_used = True
        with st.spinner("Маршрутизация через PostGIS/pgRouting..."):
            res = route_between_db(engine, network_type, start, end, weight=weight)
        if not res.ok or res.gdf is None:
            errors.append(res.error or "DB routing failed")
        else:
            gdf = res.gdf
            try:
                gdf = gdf.rename_geometry("geometry")
            except Exception:
                pass
            routes = [gdf]

            edges_table = f"edges_{network_type.lower()}"
            edge_ids = [int(x) for x in gdf["id"].tolist()] if "id" in gdf.columns else []
            route_totals = _route_totals_from_edges_table(engine, edges_table, edge_ids)
    else:
        with st.spinner("Загрузка графа и построение маршрута..."):
            try:
                routes, errors = build_routes(network_type, latlons, weight=weight)
            except Exception as e:
                errors = [str(e)]
                routes = []

with left_col:
    st.divider()

    if start and end and start == end:
        st.warning("Старт и финиш совпадают.")

    if errors:
        st.warning("Не получилось построить маршрут:\n" + "\n".join(errors))
        st.info("Попробуй увеличить радиус или сменить тип сети (walk/drive).")

    TIME_SCALE = {"walk": 14.5, "bike": 5.35, "drive": 1.85}

    if db_used and route_totals is not None and not errors:
        dist_m, time_s = route_totals

        scale = TIME_SCALE.get(network_type, 1.0)
        time_s_display = time_s * scale

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Расстояние", _format_distance_m(dist_m))
        with m2:
            st.metric("Время", _format_duration_s(time_s_display))

with right_col:
    if not start or not end:
        st.info("Введи старт и финиш и нажми Enter в каждой строке.")
    else:
        try:
            if start == end:
                m = build_map([start, end], [], names)
            else:
                m = build_map(latlons, routes, names)

            components.v1.html(m.get_root().render(), height=720, scrolling=False)
        except Exception as e:
            st.error(f"Ошибка при отображении карты:\n\n{e}")
