from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import os
import numpy as np
import geopandas as gpd
import pandas as pd
import osmnx as ox
from sqlalchemy import create_engine, text, inspect


@dataclass(frozen=True)
class SegmentResult:
    ok: bool
    error: str | None = None
    gdf: Any = None  # GeoDataFrame


def _get_engine():
    url = os.getenv("POSTGIS_URL", "").strip()
    if not url:
        return None
    return create_engine(url, pool_pre_ping=True)


def init_extensions(engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgrouting;"))


def _tables(network_type: str) -> tuple[str, str]:
    # по одному набору таблиц на walk/bike/drive (фиксированная область)
    nt = network_type.lower()
    if nt not in {"walk", "bike", "drive"}:
        raise ValueError(f"Unsupported network_type: {network_type}")
    return f"nodes_{nt}", f"edges_{nt}"


def graph_in_db(engine, network_type: str) -> bool:
    nodes_t, edges_t = _tables(network_type)
    insp = inspect(engine)
    if not insp.has_table(nodes_t) or not insp.has_table(edges_t):
        return False

    with engine.begin() as conn:
        n = conn.execute(text(f"SELECT COUNT(*) FROM {nodes_t};")).scalar_one()
        e = conn.execute(text(f"SELECT COUNT(*) FROM {edges_t};")).scalar_one()
    return (n > 0) and (e > 0)


def import_graph_to_db(engine, network_type: str, G) -> None:
    """
    Импорт networkx MultiDiGraph (OSMnx) -> PostGIS таблицы, пригодные для pgRouting.

    Схема:
      nodes_*: id BIGINT, geom POINT(4326)
      edges_*: id BIGSERIAL-like, source BIGINT, target BIGINT,
               cost_len DOUBLE, rcost_len DOUBLE,
               cost_time DOUBLE, rcost_time DOUBLE,
               geom LINESTRING/MULTILINESTRING(4326)
    """
    nodes_t, edges_t = _tables(network_type)

    # OSMnx -> GeoDataFrames
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G, nodes=True, edges=True, fill_edge_geometry=True)

    # nodes
    idx_name = gdf_nodes.index.name or "index"
    nodes = gdf_nodes.reset_index().rename(columns={idx_name: "id"}).copy()

    # на всякий случай: гарантируем POINT в 4326
    nodes = nodes[["id", "geometry"]].rename_geometry("geom")
    nodes = nodes.set_crs("EPSG:4326", allow_override=True)

    # edges
    edges = gdf_edges.reset_index().copy()  # даст u,v,key
    # стабильный id для рёбер (pgRouting любит простой int id)
    edges = edges.reset_index(drop=True)
    edges["id"] = (edges.index.astype(np.int64) + 1)

    # source/target — это node ids из OSMnx (u/v)
    edges["source"] = edges["u"].astype(np.int64)
    edges["target"] = edges["v"].astype(np.int64)

    # costs
    if "length" not in edges:
        raise ValueError("Edges do not have 'length' column.")
    if "travel_time" not in edges:
        raise ValueError("Edges do not have 'travel_time' column (call ensure_travel_time before import).")

    edges["cost_len"] = edges["length"].astype(float)
    edges["cost_time"] = edges["travel_time"].astype(float)

    # directed граф: в обратную сторону ходить нельзя через reverse_cost
    # если существует обратное ребро, оно будет отдельной строкой
    edges["rcost_len"] = -1.0
    edges["rcost_time"] = -1.0

    edges = gpd.GeoDataFrame(edges, geometry="geometry", crs="EPSG:4326").rename_geometry("geom")
    edges = edges[["id", "source", "target", "cost_len", "rcost_len", "cost_time", "rcost_time", "geom"]]

    # запись
    init_extensions(engine)
    # replace проще для учебного проекта: фиксированная область, один граф на тип сети
    nodes.to_postgis(nodes_t, engine, if_exists="replace", index=False)
    edges.to_postgis(edges_t, engine, if_exists="replace", index=False)

    # индексы
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {nodes_t} ADD PRIMARY KEY (id);"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS {nodes_t}_geom_gix ON {nodes_t} USING GIST (geom);"))

        conn.execute(text(f"CREATE INDEX IF NOT EXISTS {edges_t}_geom_gix ON {edges_t} USING GIST (geom);"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS {edges_t}_src_idx ON {edges_t} (source);"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS {edges_t}_tgt_idx ON {edges_t} (target);"))
        conn.execute(text(f"ANALYZE {nodes_t};"))
        conn.execute(text(f"ANALYZE {edges_t};"))


def route_between_db(engine, network_type: str, start: tuple[float, float], end: tuple[float, float], weight: str) -> SegmentResult:
    nodes_t, edges_t = _tables(network_type)

    if weight not in {"length", "travel_time"}:
        return SegmentResult(ok=False, error=f"Unsupported weight: {weight}")

    # маппинг cost колонок
    cost_col = "cost_time" if weight == "travel_time" else "cost_len"
    rcost_col = "rcost_time" if weight == "travel_time" else "rcost_len"

    sql = f"""
    WITH
      s AS (
        SELECT id
        FROM {nodes_t}
        ORDER BY geom <-> ST_SetSRID(ST_MakePoint(:slon, :slat), 4326)
        LIMIT 1
      ),
      t AS (
        SELECT id
        FROM {nodes_t}
        ORDER BY geom <-> ST_SetSRID(ST_MakePoint(:elon, :elat), 4326)
        LIMIT 1
      ),
      r AS (
        SELECT *
        FROM pgr_dijkstra(
          'SELECT id, source, target, {cost_col} AS cost, {rcost_col} AS reverse_cost FROM {edges_t}',
          (SELECT id FROM s),
          (SELECT id FROM t),
          directed := true
        )
      )
    SELECT e.id, e.{cost_col} AS cost, e.geom
    FROM r
    JOIN {edges_t} e ON e.id = r.edge
    WHERE r.edge <> -1
    ORDER BY r.seq;
    """

    try:
        gdf = gpd.read_postgis(
            text(sql),
            con=engine,
            geom_col="geom",
            params={"slat": start[0], "slon": start[1], "elat": end[0], "elon": end[1]},
        )
        if gdf.empty:
            return SegmentResult(ok=False, error="No path between points (empty result)")
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        return SegmentResult(ok=True, gdf=gdf)
    except Exception as e:
        return SegmentResult(ok=False, error=str(e))


def get_postgis_engine_or_none():
    return _get_engine()
