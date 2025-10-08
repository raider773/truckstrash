import json
from shapely import wkt

def load_graph_json(filepath):
    """
    Loads a graph saved in JSON and reconstructs:
      - nodes: node_id -> {"coord": (lon, lat), "type": "intersection/bin"}
      - edges: edge_id -> {"u": node_id, "v": node_id, "street_name", "geometry", ...}
      - street_graph: adjacency list node_id -> [(neighbor_id, edge_id), ...]
    
    Args:
        filepath (str): path to the saved JSON file
    
    Returns:
        nodes, edges, street_graph
    """
    
    with open(filepath, "r") as f:
        data = json.load(f)

    # Reconstruct Nodes
    nodes_loaded = {int(k): v for k, v in data["nodes"].items()}

    # Reconstruct Edges
    edges_loaded = {}
    for k, v in data["edges"].items():
        edge_id = int(k)
        edge_data = v.copy()
        if "geometry" in edge_data and edge_data["geometry"]:
            edge_data["geometry"] = wkt.loads(edge_data["geometry"])
        edges_loaded[edge_id] = edge_data

    # Reconstruct street Graph
    street_graph_loaded = {int(k): v for k, v in data["graph"].items()}

    return nodes_loaded, edges_loaded, street_graph_loaded