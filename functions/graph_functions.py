import numpy as np
from shapely.geometry import Point,LineString
import itertools
import random


def create_counters(nodes, edges):
    """
    Crea contadores para IDs de nodos y aristas.
    Esto se llama una vez al inicio, cuando ya existen nodes y edges.
    """
    node_id_counter = itertools.count(max(nodes.keys()) + 1 if nodes else 1)
    edge_id_counter = itertools.count(max(edges.keys()) + 1 if edges else 1)
    return node_id_counter, edge_id_counter


def find_closest_edge(bin_coord, edges, max_distance=float('inf')):
    """
    Encuentra la arista más cercana a un punto (bin_coord).
    """
    bin_point = Point(bin_coord)
    closest_edge_id = None
    min_dist = float('inf')

    for edge_id, data in edges.items():
        edge_geom = data["geometry"]
        dist = edge_geom.distance(bin_point)
        if dist < min_dist:
            min_dist = dist
            closest_edge_id = edge_id

    if min_dist > max_distance:
        return None
    return closest_edge_id


def insert_bin_into_edge(bin_id, bin_coord, edge_id, nodes, edges, street_graph,
                         edge_id_counter, priority=1):
    """
    Inserta un nodo tipo 'bin' en una arista existente, actualizando el grafo.
    
    Args:
        bin_id (int): ID del bin (ya creado previamente)
        bin_coord (tuple): Coordenadas (lon, lat)
        edge_id (int): ID de la arista en la que se inserta
        nodes (dict): Nodos existentes
        edges (dict): Aristas existentes
        street_graph (dict): Grafo de adyacencia
        edge_id_counter (itertools.count): contador para IDs de aristas nuevas
        priority (int): prioridad del bin
    """
    # Añadir el nodo bin
    nodes[bin_id] = {"coord": bin_coord, "type": "bin", "priority": priority}

    old_edge = edges[edge_id]
    u, v = old_edge["u"], old_edge["v"]

    # Eliminar el edge original del grafo
    if (v, edge_id) in street_graph.get(u, []):
        street_graph[u].remove((v, edge_id))
    if (u, edge_id) in street_graph.get(v, []):
        street_graph[v].remove((u, edge_id))

    # Crear dos nuevos edges (u -> bin, bin -> v)
    edge1_id = next(edge_id_counter)
    edge2_id = next(edge_id_counter)

    edge1 = old_edge.copy()
    edge1.update({"u": u, "v": bin_id})
    edges[edge1_id] = edge1

    edge2 = old_edge.copy()
    edge2.update({"u": bin_id, "v": v})
    edges[edge2_id] = edge2

    # Actualizar adyacencias según dirección
    direction = old_edge.get("direction", "doble").lower()
    if direction == "creciente":
        street_graph.setdefault(u, []).append((bin_id, edge1_id))
        street_graph.setdefault(bin_id, []).append((v, edge2_id))
    elif direction == "decreciente":
        street_graph.setdefault(bin_id, []).append((u, edge1_id))
        street_graph.setdefault(v, []).append((bin_id, edge2_id))
    else:  # bidireccional
        street_graph.setdefault(u, []).append((bin_id, edge1_id))
        street_graph.setdefault(bin_id, []).append((u, edge1_id))
        street_graph.setdefault(bin_id, []).append((v, edge2_id))
        street_graph.setdefault(v, []).append((bin_id, edge2_id))

    return bin_id


def add_bin(bin_id, bin_coord, nodes, edges, street_graph,
            edge_id_counter, max_distance=20, priority=1):
    """
    Añade un bin al grafo con un ID ya existente y prioridad.

    Args:
        bin_id (int): ID del bin (ya creado)
        bin_coord (tuple): coordenadas (lon, lat)
        nodes (dict): nodos existentes
        edges (dict): aristas existentes
        street_graph (dict): grafo de adyacencia
        edge_id_counter (itertools.count): contador para nuevos edges
        max_distance (float): distancia máxima para snap al edge
        priority (int/float): prioridad del bin

    Returns:
        int: el ID del bin insertado, o None si no hay edge cercano
    """
    edge_id = find_closest_edge(bin_coord, edges, max_distance)
    if edge_id is None:
        print(f"No edge close enough to place the bin at {bin_coord}.")
        return None

    return insert_bin_into_edge(
        bin_id, bin_coord, edge_id,
        nodes, edges, street_graph,
        edge_id_counter,
        priority=priority
    )


import random
import numpy as np

def generate_random_bin_coords(neighborhood, edges, nodes, num_bins=1, min_id=0):
    """
    Generate random bin coordinates with unique IDs based on the current nodes,
    starting from a specified minimum ID.

    Args:
        neighborhood (int): neighborhood ID to filter edges
        edges (dict): edge_id -> edge attributes (including 'geometry' and 'neighborhood')
        nodes (dict): existing nodes (used to find next available IDs)
        num_bins (int): number of bins to generate
        min_id (int): minimum ID allowed for bins (e.g. 17718)

    Returns:
        List of dicts with 'id', 'coord', and 'neighborhood'
    """
    # Filter edges in the neighborhood
    neighborhood_edges = [
        data["geometry"] for data in edges.values()
        if not np.isnan(data.get("neighborhood", np.nan))
        and int(data["neighborhood"]) == neighborhood
    ]

    if not neighborhood_edges:
        raise ValueError(f"No edges found in neighborhood {neighborhood}")

    # Start ID based on the max node ID and min_id parameter
    start_id = max(max(nodes.keys()), min_id - 1) + 1 if nodes else min_id
    bins = []

    for i in range(num_bins):
        edge_geom = random.choice(neighborhood_edges)
        # Random point along the edge (not always the midpoint)
        t = random.random()
        point = edge_geom.interpolate(t, normalized=True)

        bin_id = start_id + i
        bins.append({
            "id": bin_id,
            "coord": (point.x, point.y),
            "neighborhood": neighborhood
        })

    return bins

import numpy as np
import random

def get_depot_info(depot_id, comuna, nodes, edges):
    """
    Get information about a given depot node within a specific neighborhood (comuna).

    Args:
        depot_id (int): ID of the depot node (must exist in the graph)
        comuna (int): neighborhood (comuna) ID
        nodes (dict): node_id -> node attributes
        edges (dict): edge_id -> edge attributes (must include 'neighborhood')

    Returns:
        dict: Information about the depot node {'id', 'coord', 'neighborhood'}
    """
    # Validate that depot_id exists
    if depot_id not in nodes:
        raise ValueError(f"Depot ID {depot_id} not found in nodes.")

    # Find all edges belonging to the given neighborhood
    edges_in_neigh = [
        edge_id for edge_id, data in edges.items()
        if not np.isnan(data.get('neighborhood', np.nan))
        and int(data['neighborhood']) == comuna
    ]

    if not edges_in_neigh:
        raise ValueError(f"No edges found in neighborhood {comuna}")

    # Optional: verify depot is inside that neighborhood
    in_neigh = False
    for edge_id in edges_in_neigh:
        u, v = edges[edge_id]['u'], edges[edge_id]['v']
        if depot_id in (u, v):
            in_neigh = True
            break

    if not in_neigh:
        print(f"Depot node {depot_id} not directly connected to neighborhood {comuna} edges.")

    depot_info = {
        "id": depot_id,
        "coord": nodes[depot_id]['coord'],
        "neighborhood": comuna
    }

    print(f"Selected depot node ID: {depot_id}, coordinates: {nodes[depot_id]['coord']}")
    return depot_info

def plan_truck_routes(nodes, street_graph, edges, num_trucks, depot_id, balance_factor=1.0):
    """
    Plan routes for multiple trucks to collect bins using a priority-weighted greedy
    algorithm with 2-opt improvement and load balancing.

    Args:
        nodes (dict): Dictionary of all nodes in the graph.
                      Each node must have a 'type' key ('intersection' or 'bin')
                      and optionally a 'priority' key for bins.
        street_graph (dict): Adjacency list of the graph {node_id: [(neighbor_id, edge_id), ...]}
        edges (dict): Dictionary of edges with weights (e.g., 'distance').
        num_trucks (int): Number of trucks available for collection.
        depot_id (int): Node ID of the truck depot (starting/ending point).
        balance_factor (float): Weight for load balancing. Higher = more balanced routes.

    Returns:
        routes (list of lists): Each truck's route as a list of node IDs, starting and ending at depot.
        distance_matrix (dict of dicts): Precomputed shortest distances between nodes of interest.
        path_matrix (dict): Previous-node dictionaries from Dijkstra for path reconstruction.
    """

    import heapq

    # -------------------------------------------------------------
    # Step 0: Define helper functions
    # -------------------------------------------------------------

    # Dijkstra shortest path for directed graph
    def dijkstra_directed(graph, edges, start):
        """
        Compute shortest paths from start node to all others in the graph.
        Returns distances and previous-node mapping.
        """
        dist = {start: 0}  # distance from start to each node
        prev = {start: None}  # previous node for path reconstruction
        pq = [(0, start)]  # priority queue

        while pq:
            d_curr, u = heapq.heappop(pq)
            if d_curr > dist[u]:
                continue  # already found better path
            for v, edge_id in graph.get(u, []):
                weight = edges[edge_id].get("distance", 1)  # default weight 1
                alt = d_curr + weight
                if v not in dist or alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
                    heapq.heappush(pq, (alt, v))
        return dist, prev

    # Reconstruct path from u -> v using previous-node mapping
    def reconstruct_path(prev, u, v):
        path = []
        current = v
        while current is not None:
            path.append(current)
            current = prev.get(current)
        path = path[::-1]  # reverse
        if path[0] != u:
            return []  # no path found
        return path

    # 2-opt optimization for improving a route
    def two_opt(route, distance_matrix):
        """
        Apply 2-opt swap to reduce route distance.
        route: list of node IDs
        distance_matrix: dict of distances between nodes
        """
        best = route[:]
        improved = True
        while improved:
            improved = False
            n = len(best)
            for i in range(1, n-2):
                for j in range(i+1, n-1):
                    if j - i == 1:
                        continue  # skip adjacent nodes
                    a, b = best[i-1], best[i]
                    c, d = best[j], best[j+1]
                    old = distance_matrix[a][b] + distance_matrix[c][d]
                    new = distance_matrix[a][c] + distance_matrix[b][d]
                    if new < old:
                        best[i:j+1] = reversed(best[i:j+1])
                        improved = True
            route = best
        return best

    # -------------------------------------------------------------
    # Step 1: Select all bins and sort by priority
    # -------------------------------------------------------------
    bins = [nid for nid, data in nodes.items() if data['type'] == 'bin']
    if not bins:
        # no bins to collect, return empty routes
        return [[] for _ in range(num_trucks)], {}, {}

    # Sort bins by priority (highest first)
    bins_sorted = sorted(bins, key=lambda nid: nodes[nid].get('priority', 1), reverse=True)

    # -------------------------------------------------------------
    # Step 2: Precompute distance matrix between depot + bins
    # -------------------------------------------------------------
    nodes_of_interest = [depot_id] + bins_sorted
    distance_matrix = {}  # distance_matrix[u][v] = shortest distance from u to v
    path_matrix = {}      # path_matrix[u] = previous-node mapping from Dijkstra

    for u in nodes_of_interest:
        dist, prev = dijkstra_directed(street_graph, edges, u)
        distance_matrix[u] = {v: dist[v] for v in nodes_of_interest if v != u and v in dist}
        path_matrix[u] = prev

    # -------------------------------------------------------------
    # Step 3: Initialize empty routes for each truck (starting at depot)
    # -------------------------------------------------------------
    routes = [[depot_id] for _ in range(num_trucks)]

    # -------------------------------------------------------------
    # Step 4: Assign bins to trucks using priority-weighted greedy
    # -------------------------------------------------------------
    for bin_id in bins_sorted:
        best_truck = None
        best_score = float('inf')
        for t_idx in range(num_trucks):
            last_node = routes[t_idx][-1]
            if bin_id not in distance_matrix[last_node]:
                continue  # cannot reach this bin from current truck
            dist_cost = distance_matrix[last_node][bin_id]
            load_penalty = balance_factor * (len(routes[t_idx]) - 1)  # penalize trucks with more bins
            score = dist_cost + load_penalty
            if score < best_score:
                best_score = score
                best_truck = t_idx
        if best_truck is None:
            raise ValueError(f"No path found from any truck to bin {bin_id}")
        routes[best_truck].append(bin_id)

    # -------------------------------------------------------------
    # Step 5: Close routes by returning to depot
    # -------------------------------------------------------------
    for t_idx in range(num_trucks):
        routes[t_idx].append(depot_id)

    # -------------------------------------------------------------
    # Step 6: Optional 2-opt improvement for each truck's route
    # -------------------------------------------------------------
    for t_idx in range(num_trucks):
        route = routes[t_idx]
        if len(route) > 3:  # must have at least depot + 2 nodes
            routes[t_idx] = two_opt(route, distance_matrix)

    # -------------------------------------------------------------
    # Step 7: Return routes and precomputed matrices
    # -------------------------------------------------------------
    return routes, distance_matrix, path_matrix




