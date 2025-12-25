from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import folium
import numpy as np
import osmnx as ox


def configure_osmnx() -> None:
    ox.settings.use_cache = True
    ox.settings.log_console = False

    ox.settings.requests_timeout = 60 
    ox.settings.overpass_settings = "[out:json][timeout:60]"


@dataclass(frozen=True)
class SegmentResult:
    ok: bool
    error: str | None = None
    gdf: Any = None # type: ignore


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


def load_graph_from_point(
    center_latlon: tuple[float, float],
    network_type: str,
    dist_m: int,
    use_travel_time: bool,
):
    configure_osmnx()

    G = ox.graph_from_point(
        center_latlon,
        dist=dist_m,
        network_type=network_type,
        simplify=True,
    )

    if use_travel_time:
        G = ensure_travel_time(G)

    return G


def route_between(G, start: tuple[float, float], end: tuple[float, float], weight: str) -> SegmentResult:
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


def build_routes(G, places_latlon: list[tuple[float, float]], weight: str) -> tuple[list, list[str]]:
    routes = []
    errors: list[str] = []

    for i in range(len(places_latlon) - 1):
        a = places_latlon[i]
        b = places_latlon[i + 1]
        res = route_between(G, a, b, weight=weight)
        if res.ok:
            routes.append(res.gdf)
        else:
            errors.append(f"Segment {i + 1}: {res.error or 'unknown error'}")

    return routes, errors


def build_map(
    places_latlon: list[tuple[float, float]],
    routes_gdf: Iterable,
    place_names: list[str],
    zoom_start: int = 13,
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
