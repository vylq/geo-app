from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import folium
import numpy as np
import osmnx as ox

from postgis_engine import (
    SegmentResult as DbSegmentResult,
    get_postgis_engine_or_none,
    graph_in_db,
    import_graph_to_db,
    route_between_db,
)

BASE_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = BASE_DIR / "cache"
GRAPH_CACHE_DIR = CACHE_DIR / "graphs"
OSMNX_CACHE_DIR = CACHE_DIR / "osmnx"

FIXED_CENTER_LATLON = (59.849224, 30.144109)
FIXED_DIST_M = 20_000


def configure_osmnx() -> None:
    OSMNX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ox.settings.cache_folder = str(OSMNX_CACHE_DIR)
    ox.settings.use_cache = True
    ox.settings.log_console = False

    ox.settings.requests_timeout = 60
    ox.settings.overpass_settings = "[out:json][timeout:60]"


@dataclass(frozen=True)
class SegmentResult:
    ok: bool
    error: str | None = None
    gdf: Any = None  # type: ignore


def safe_mean_center(latlons: list[tuple[float, float]]) -> tuple[float, float]:
    if not latlons:
        return 52.5200, 13.4050
    lat = float(np.mean([p[0] for p in latlons]))
    lon = float(np.mean([p[1] for p in latlons]))
    return lat, lon


def ensure_travel_time(G):
    try:
        any_edge = next(iter(G.edges(data=True)))[2]
    except StopIteration:
        return G

    if "travel_time" in any_edge:
        return G

    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    return G


def load_fixed_graph_local(network_type: str):
    """
    ТОЛЬКО локальный GraphML кэш + скачивание OSM при отсутствии.
    (То есть как у тебя, но чуть явно названо.)
    """
    configure_osmnx()
    GRAPH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_name = f"{network_type}_{FIXED_DIST_M}m.graphml"
    cache_path = GRAPH_CACHE_DIR / cache_name

    if cache_path.exists():
        G = ox.load_graphml(cache_path)
        if G.number_of_edges() == 0:
            cache_path.unlink(missing_ok=True)
        else:
            G = ensure_travel_time(G)
            # обновим кэш, если travel_time добавили
            ox.save_graphml(G, cache_path)
            return G

    # если файла нет — качаем из OSM (центр/радиус фиксированные)
    G = ox.graph_from_point(
        FIXED_CENTER_LATLON,
        dist=FIXED_DIST_M,
        network_type=network_type,
        simplify=True,
    )
    G = ensure_travel_time(G)
    if G.number_of_edges() == 0:
        raise ValueError("OSM returned a graph with no edges for the fixed area.")

    ox.save_graphml(G, cache_path)
    return G


def ensure_graph_in_postgis(network_type: str) -> None:
    """
    Гарантирует, что нужный граф есть в PostGIS.

    Приоритет как ты просил:
      1) если локальный graphml есть — используем его как источник
      2) если локального нет — качаем через OSMnx и сохраняем локально
    """
    engine = get_postgis_engine_or_none()
    if engine is None:
        return

    if graph_in_db(engine, network_type):
        return

    # источник для импорта: локальный файл (если есть), иначе скачка
    G = load_fixed_graph_local(network_type)
    import_graph_to_db(engine, network_type, G)


def route_between_local(G, start: tuple[float, float], end: tuple[float, float], weight: str) -> SegmentResult:
    try:
        start_node = ox.distance.nearest_nodes(G, X=start[1], Y=start[0])
        end_node = ox.distance.nearest_nodes(G, X=end[1], Y=end[0])
        path = ox.routing.shortest_path(G, start_node, end_node, weight=weight)

        if path is None:
            return SegmentResult(ok=False, error="No path between points")

        gdf = ox.routing.route_to_gdf(G, path, weight=weight)
        return SegmentResult(ok=True, gdf=gdf)

    except Exception as e:
        return SegmentResult(ok=False, error=str(e))


def build_routes(network_type: str, places_latlon: list[tuple[float, float]], weight: str) -> tuple[list, list[str]]:
    """
    Если задан POSTGIS_URL -> роутинг в PostGIS (pgRouting).
    Иначе -> как раньше локально через OSMnx/NX.
    """
    routes = []
    errors: list[str] = []

    engine = get_postgis_engine_or_none()
    use_db = engine is not None

    if use_db:
        ensure_graph_in_postgis(network_type)

    G = None
    if not use_db:
        G = load_fixed_graph_local(network_type)

    for i in range(len(places_latlon) - 1):
        a = places_latlon[i]
        b = places_latlon[i + 1]

        if use_db:
            res: DbSegmentResult = route_between_db(engine, network_type, a, b, weight=weight)  # type: ignore[arg-type]
            if res.ok:
                routes.append(res.gdf)
            else:
                errors.append(f"Segment {i + 1}: {res.error or 'unknown error'}")
        else:
            res = route_between_local(G, a, b, weight=weight)  # type: ignore[arg-type]
            if res.ok:
                routes.append(res.gdf)
            else:
                errors.append(f"Segment {i + 1}: {res.error or 'unknown error'}")

    return routes, errors


def build_map(
    places_latlon: list[tuple[float, float]],
    routes_gdf: Iterable,
    place_names: list[str],
    zoom_start: int = 11,
):
    center = safe_mean_center(places_latlon)

    m = folium.Map(
        location=center,
        tiles="Cartodb Positron",
        zoom_start=zoom_start,
        control_scale=True,
    )

    for route in routes_gdf:
        folium.GeoJson(route).add_to(m)

    for i, (lat, lon) in enumerate(places_latlon):
        if i == 0:
            icon = folium.Icon(prefix="fa", icon="person", color="green")
        elif i == len(places_latlon) - 1:
            icon = folium.Icon(prefix="fa", icon="flag-checkered", color="red")
        else:
            icon = folium.Icon(prefix="fa", icon="star", color="orange")

        label = place_names[i] if i < len(place_names) else f"Point {i}"
        folium.Marker(
            location=(lat, lon),
            popup=label,
            tooltip=label,
            icon=icon,
        ).add_to(m)

    return m
