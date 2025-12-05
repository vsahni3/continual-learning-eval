import requests
import random
import json
import time
from datetime import datetime, timedelta
import names
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import os
from dotenv import load_dotenv
load_dotenv()
import threading

GLOBAL_USED_NAMES = set()
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

def retry(fn: callable, *args, **kwargs):
    for i in range(10):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            safe_print(f"[{datetime.now()}] retry {i+1}/10: {e}")
            time.sleep(1 + i*0.3)
    raise RuntimeError("Failed after 10 retries")

def prompt_gpt(prompt: str, is_staging: bool = False) -> str:
    url = f"https://{'stg.' if is_staging else ''}api.cohere.ai/compatibility/v1/chat/completions"
    response = requests.post(url, headers={
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"bearer {os.getenv('COHERE_API_KEY')}"
    }, json={
        "model": "command-a-03-2025",
        "messages": [{"role": "user", "content": prompt}]
    }, timeout=2000)
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

def clean_json_response(response: str) -> str:
    """Strip markdown code blocks from LLM response"""
    response = response.strip()
    if response.startswith("```"):
        lines = response.split("\n")
        response = "\n".join([l for l in lines if not l.startswith("```")])
    return response

def generate_person_nodes(num_people: int = 100) -> list[dict]:
    people = []
    for i in range(num_people):
        full_name = None
        while full_name is None or full_name in GLOBAL_USED_NAMES:
            full_name = names.get_full_name()
        GLOBAL_USED_NAMES.add(full_name)
        people.append({"id": f"person_{i}", "name": full_name, "type": "person"})
    safe_print(f"[{datetime.now()}] Generated {num_people} person nodes")
    return people

def generate_typed_nodes(num: int, node_type: str, examples: str) -> list[dict]:
    """Generic function to generate location or event nodes"""
    def inner() -> list[dict]:
        response = prompt_gpt(f"""Generate {num} diverse {node_type} names for a social graph simulation.
{examples}

Return ONLY a JSON array: [{{"name": "Name", "type": "subtype"}}, ...]
Do not use markdown code blocks.""")

        data = json.loads(clean_json_response(response))
        nodes = [{
            "id": f"{node_type}_{i}",
            "name": item["name"],
            "subtype": item["type"],
            "type": node_type
        } for i, item in enumerate(data)]

        safe_print(f"[{datetime.now()}] Generated {len(nodes)} {node_type} nodes")
        return nodes
    return retry(inner)

def generate_location_nodes(num: int = 50) -> list[dict]:
    return generate_typed_nodes(num, "location",
        "Mix of cities, venues, institutions, places")

def generate_event_nodes(num: int = 100) -> list[dict]:
    return generate_typed_nodes(num, "event",
        "Mix of concerts, conferences, social, sports, cultural events")

def get_context(node_id: str, all_edges: list[dict], all_nodes_dict: dict, depth: int = 1) -> dict:
    """Get context for a node within N hops (BFS)"""
    edges_by_node = {}
    for edge in all_edges:
        edges_by_node.setdefault(edge["src"], []).append(edge)
        edges_by_node.setdefault(edge["dst"], []).append(edge)

    connected_edges = []
    visited = set()
    current_layer = {node_id}

    for _ in range(depth):
        next_layer = set()
        for node in current_layer:
            if node in visited:
                continue
            visited.add(node)
            for edge in edges_by_node.get(node, []):
                # print(edge["dst"] if edge["src"] == node else edge["src"])
                if edge not in connected_edges:
                    connected_edges.append(edge)
                next_layer.add(edge["dst"] if edge["src"] == node else edge["src"])
        current_layer = next_layer

    # visited.remove(node_id)
    connected_nodes = [all_nodes_dict[nid] for nid in visited.union(current_layer) - {node_id}]

    # Format context
    lines = [f"Context for {all_nodes_dict[node_id]['name']} at depth {depth}:"]
    if not connected_edges:
        lines.append("  No existing connections")
    else:
        lines.append(f"  {len(connected_edges)} edges, {len(connected_nodes)} nodes:")
        for edge in connected_edges:
            src = all_nodes_dict[edge["src"]]["name"]
            dst = all_nodes_dict[edge["dst"]]["name"]
            relation = edge.get('relation', '')
            info = f"{src} --[{edge['type']}]--> {dst}"
            if relation:
                info += f" ({relation})"
            info += f" [{edge['start_date']} to {edge['end_date']}]"
            lines.append(f"    {info}")

    return {
        "connected_nodes": connected_nodes,
        "connected_edges": connected_edges,
        "context_text": "\n".join(lines)
    }

EDGE_RULES = """
EDGE TYPES & RULES:

1. person_to_person:
   - Valid: friend, colleague, boss, employee, mentor, partner, cousin, in-law
   - FORBIDDEN: parent, child, sibling, grandparent, grandchild (require matching surnames)
   - NOTE: Some relationships (e.g., 'married', 'partner') may require corresponding event nodes (e.g., wedding event)

2. person_to_location:
   - Valid: lives_at, works_at, visited, frequent_visitor

3. person_to_event:
   - Valid: attended, organized, performed, spoke

4. event_to_location:
   - Valid: held_at

CONSISTENCY RULES:
- Check existing contexts to avoid contradictions
- Don't create duplicate edges between same nodes
- Ensure relationship types are compatible with existing connections
- Some edges may imply the existence of event nodes (e.g., marriage relationship requires wedding event)

DURATION RULES - duration_days must match the relation type:
- Events (1-3 days): attended, organized, performed, spoke, held_at, visited
- Medium-term (7-90 days): frequent_visitor, temporary colleague/mentor relationships
- Long-term (90+ days): lives_at, works_at, friend, partner, boss, employee, family relations
- NEVER assign long durations to event edges or short durations to stable relationships
"""

def generate_edges_batch(sampled_nodes: list[dict], contexts: dict[str, dict],
                         current_date: str, target_edges: int,
                         max_duration: int) -> list[dict]:
    """Generate multiple edges in a single LLM call"""

    def inner() -> list[dict]:
        # Format nodes list
        nodes_section = "NODES:\n"
        contexts_section = "\nCONTEXTS:\n"
        for i, node in enumerate(sampled_nodes, 1):
            node_subtype = node.get("subtype", "")
            subtype_str = f", type: {node_subtype}" if node_subtype else ""
            nodes_section += f"{i}. {node['id']}: {node['name']} ({node['type']}{subtype_str})\n"
            contexts_section += f"\n{contexts[node['id']]['context_text']}\n"


        prompt = f"""You have {len(sampled_nodes)} nodes in a social graph for date {current_date}.

{nodes_section}
{contexts_section}

TASK: Generate EXACTLY {target_edges} edges between these nodes.

{EDGE_RULES}

Return ONLY a JSON array (no markdown):
[
  {{
    "src": "node_id",
    "dst": "node_id",
    "type": "person_to_person|person_to_location|person_to_event|event_to_location",
    "relation": "relationship type (e.g., friend, works_at, attended, held_at)",
    "duration_days": 1-{max_duration}
  }},
  ...
]

Generate diverse, realistic connections. Ensure internal consistency."""

        response = clean_json_response(prompt_gpt(prompt))
        edges_data = json.loads(response)

        # Validate and convert to edge objects
        valid_node_ids = {n["id"] for n in sampled_nodes}
        edges = []

        for edge_data in edges_data:
            # Validate node IDs
            if edge_data["src"] not in valid_node_ids or edge_data["dst"] not in valid_node_ids:
                safe_print(f"[{datetime.now()}] Skipping invalid node IDs: {edge_data['src']} -> {edge_data['dst']}")
                continue

            # Calculate end date
            duration = min(edge_data["duration_days"], max_duration)
            end_date = (datetime.strptime(current_date, "%Y-%m-%d") +
                       timedelta(days=duration - 1)).strftime("%Y-%m-%d")

            edge = {
                "type": edge_data["type"],
                "src": edge_data["src"],
                "dst": edge_data["dst"],
                "start_date": current_date,
                "end_date": end_date,
                "relation": edge_data.get("relation", "")
            }
            edges.append(edge)

            safe_print(f"[{datetime.now()}] Created: {edge_data['src']} --[{edge_data['type']}]--> {edge_data['dst']}")

        safe_print(f"[{datetime.now()}] Batch generated {len(edges)} edges")
        return edges

    return retry(inner)

def ensure_node_consistency(newly_generated_edges: list[dict], all_nodes_dict: dict,
                            contexts: dict[str, dict], current_date: str,
                            max_duration: int, all_edges: list[dict]) -> tuple[list[dict], list[dict]]:
    """Use LLM to identify and generate any missing nodes needed for edge consistency.

    Args:
        newly_generated_edges: List of edges just generated
        all_nodes_dict: Dictionary of all existing nodes {id: node}
        contexts: Dictionary of contexts for nodes involved in edges
        current_date: Current date for new nodes
        max_duration: Maximum duration for new edges in days
        all_edges: All existing edges in the graph

    Returns:
        Tuple of (new_nodes, new_edges) that were generated
    """
    if not newly_generated_edges:
        return [], []

    def inner() -> tuple[list[dict], list[dict]]:
        # Collect all unique node IDs from the new edges
        involved_node_ids = set()
        for edge in newly_generated_edges:
            involved_node_ids.add(edge["src"])
            involved_node_ids.add(edge["dst"])

        # Format edges with node context
        edges_section = "NEWLY GENERATED EDGES:\n"
        for i, edge in enumerate(newly_generated_edges, 1):
            src_node = all_nodes_dict[edge["src"]]
            dst_node = all_nodes_dict[edge["dst"]]
            edges_section += f"{i}. {src_node['name']} ({src_node['type']}) --[{edge['type']}: {edge.get('relation', 'N/A')}]--> "
            edges_section += f"{dst_node['name']} ({dst_node['type']}) [{edge['start_date']} to {edge['end_date']}]\n"

        # Get or compute context for involved nodes
        contexts_section = "\nEXISTING CONTEXT (depth 1 around involved nodes):\n"
        for node_id in involved_node_ids:
            context = contexts[node_id]
            contexts_section += f"\n{context['context_text']}\n"

        prompt = """Analyze the following edges in a temporal social graph to identify missing nodes needed for consistency.

IMPORTANT: Check the existing context carefully. If a required node already exists in the context, DO NOT create a duplicate.

%s
%s

Current date: %s

TASK: Identify any event or location nodes that SHOULD exist to support these edges but are missing.

First, check the EXISTING CONTEXT to see if the required nodes already exist. Only suggest creating new nodes if they are NOT already present in the context.

Examples of missing nodes:
- If two people have "married" or "partner" relation, there should be a wedding event they both attended
- If someone has a "started_job" relation, there might be a company/organization location
- If people have a "graduated_together" relation, there might be a graduation ceremony event

For each missing node you identify, also specify:
1. What edges should connect to this new node (reference by the edge number above)
2. The relation type for those new edges

Return ONLY a JSON object (no markdown):
{
  "missing_nodes": [
    {
      "name": "Node name",
      "type": "event|location",
      "subtype": "specific type (e.g., wedding, graduation, company)",
      "reasoning": "Why this node is needed",
      "related_edge_numbers": [1, 2],
      "new_edges": [
        {
          "src": "existing_node_id_from_original_edges OR NEW_NODE",
          "dst": "existing_node_id_from_original_edges OR NEW_NODE",
          "type": "person_to_event|person_to_location|event_to_location",
          "relation": "attended|organized|held_at|etc",
          "duration_days": 1
        }
      ]
    }
  ]
}

If NO missing nodes are needed, return: {"missing_nodes": []}

Be conservative - only suggest nodes that are clearly implied by the edges.""" % (edges_section, contexts_section, current_date)

        response = clean_json_response(prompt_gpt(prompt))
        data = json.loads(response)

        if not data.get("missing_nodes"):
            safe_print(f"[{datetime.now()}] No missing nodes identified")
            return [], []

        new_nodes = []
        new_edges = []

        # Get next available IDs
        existing_event_ids = [nid for nid in all_nodes_dict.keys() if nid.startswith("event_")]
        existing_location_ids = [nid for nid in all_nodes_dict.keys() if nid.startswith("location_")]
        next_event_id = max([int(eid.split("_")[1]) for eid in existing_event_ids] + [-1]) + 1
        next_location_id = max([int(lid.split("_")[1]) for lid in existing_location_ids] + [-1]) + 1

        for missing in data["missing_nodes"]:
            # Create new node
            node_type = missing["type"]
            if node_type == "event":
                node_id = f"event_{next_event_id}"
                next_event_id += 1
            else:  # location
                node_id = f"location_{next_location_id}"
                next_location_id += 1

            new_node = {
                "id": node_id,
                "name": missing["name"],
                "type": node_type,
                "subtype": missing["subtype"]
            }
            new_nodes.append(new_node)
            all_nodes_dict[node_id] = new_node  # Add to dict for edge creation


            safe_print(f"[{datetime.now()}] Generated missing node: {node_id} - {missing['name']} ({missing['reasoning']})")
            
            # Create edges to this new node
            for edge_data in missing.get("new_edges", []):
                # Replace NEW_NODE with actual node_id in either src or dst
                src = node_id if edge_data["src"] == "NEW_NODE" else edge_data["src"]
                dst = node_id if edge_data["dst"] == "NEW_NODE" else edge_data["dst"]

                duration = min(edge_data.get("duration_days", 1), max_duration)
                end_date = (datetime.strptime(current_date, "%Y-%m-%d") +
                           timedelta(days=duration - 1)).strftime("%Y-%m-%d")

                new_edge = {
                    "type": edge_data["type"],
                    "src": src,
                    "dst": dst,
                    "start_date": current_date,
                    "end_date": end_date,
                    "relation": edge_data.get("relation", "")
                }
                new_edges.append(new_edge)
                safe_print(f"[{datetime.now()}] Generated edge: {src} --[{edge_data['type']}]--> {dst}")

        safe_print(f"[{datetime.now()}] Consistency check: {len(new_nodes)} nodes, {len(new_edges)} edges added")
        return new_nodes, new_edges

    return retry(inner)

def build_temporal_graph(num_people: int, num_locations: int, num_events: int,
                         start_date: str, num_days: int,
                         nodes_per_day: int, edges_per_day: int,
                         max_edge_duration_days: int) -> dict:
    """Build temporal graph with day-by-day batch edge generation"""
    safe_print(f"[{datetime.now()}] Starting: {num_people} people, {num_locations} locations, {num_events} events over {num_days} days")

    # Generate all nodes
    people = generate_person_nodes(num_people)
    locations = generate_location_nodes(num_locations)
    events = generate_event_nodes(num_events)
    all_nodes = people + locations + events
  
    all_nodes_dict = {n["id"]: n for n in all_nodes}

    # Generate edges day-by-day
    all_edges = []
    temporal_snapshots = {}
    current_dt = datetime.strptime(start_date, "%Y-%m-%d")

    for day_idx in range(num_days):
        current_date = current_dt.strftime("%Y-%m-%d")
        safe_print(f"\n[{datetime.now()}] Day {day_idx + 1}/{num_days}: {current_date}")

        # Sample fewer nodes per day
        sampled_nodes = random.sample(all_nodes, min(nodes_per_day, len(all_nodes)))
        safe_print(f"[{datetime.now()}] Sampled {len(sampled_nodes)} nodes")

        # Fetch context for all sampled nodes once
        contexts = {}
        for node in sampled_nodes:
            contexts[node["id"]] = get_context(node["id"], all_edges, all_nodes_dict, depth=1)

        # Single LLM call to generate all edges for the day
        try:
            daily_edges = generate_edges_batch(
                sampled_nodes, contexts, current_date,
                edges_per_day, max_edge_duration_days
            )
            print(daily_edges, '\n\n\n')

            # Check for missing nodes needed for consistency
            try:
                new_nodes, new_edges = ensure_node_consistency(
                    daily_edges, all_nodes_dict, contexts, current_date,
                    max_edge_duration_days, all_edges
                )

                # Add new nodes to appropriate lists
                for node in new_nodes:
                    if node["type"] == "event":
                        events.append(node)
                    elif node["type"] == "location":
                        locations.append(node)
                    all_nodes.append(node)

                # Add new edges to daily edges and all edges
                daily_edges.extend(new_edges)

            except Exception as e:
                safe_print(f"[{datetime.now()}] Error in consistency check: {e}")

            all_edges.extend(daily_edges)
        except Exception as e:
            safe_print(f"[{datetime.now()}] Error generating batch: {e}")
            daily_edges = []

        temporal_snapshots[current_date] = daily_edges
        safe_print(f"[{datetime.now()}] Day {day_idx + 1} complete: {len(daily_edges)} edges")
        current_dt += timedelta(days=1)

    safe_print(f"\n[{datetime.now()}] Complete! Total edges: {len(all_edges)}")
    return {
        "nodes": {"people": people, "locations": locations, "events": events},
        "edges": all_edges,
        "temporal_snapshots": temporal_snapshots,
        "metadata": {
            "start_date": start_date,
            "num_days": num_days,
            "total_nodes": len(all_nodes),
            "total_edges": len(all_edges),
            "generation_date": datetime.now().isoformat()
        }
    }

def save_graph(graph: dict, filename: str = "temporal_graph.json"):
    with open(filename, 'w') as f:
        json.dump(graph, f, indent=2)
    print(f"Saved: {filename}")

def create_timeline(graph: dict) -> str:
    """Create human-readable timeline"""
    all_nodes = {n["id"]: n for nt in ["people", "locations", "events"]
                 for n in graph["nodes"][nt]}

    lines = ["=" * 80, "TEMPORAL GRAPH TIMELINE", "=" * 80, ""]

    for date in sorted(graph["temporal_snapshots"].keys()):
        edges = graph["temporal_snapshots"][date]
        lines.append(f"[{date}] {len(edges)} edges:")
        for e in edges:
            src = all_nodes[e["src"]]["name"]
            dst = all_nodes[e["dst"]]["name"]
            relation = e.get('relation', '')
            desc = f"  {src} --[{e['type']}]--> {dst}"
            if relation:
                desc += f" ({relation})"
            desc += f" [ends: {e['end_date']}]"
            lines.append(desc)
        lines.append("")

    lines.extend(["=" * 80, f"Total: {len(graph['edges'])} edges", "=" * 80])
    return "\n".join(lines)

def save_timeline(graph: dict, filename: str = "temporal_timeline.txt"):
    with open(filename, 'w') as f:
        f.write(create_timeline(graph))
    print(f"Saved: {filename}")

def visualize_matplotlib(graph: dict, filename: str = "temporal_graph.png"):
    """Static visualization"""
    G = nx.DiGraph()
    all_nodes = {n["id"]: n for nt in ["people", "locations", "events"]
                 for n in graph["nodes"][nt]}

    for nid, node in all_nodes.items():
        G.add_node(nid, node_type=node["type"], name=node["name"])

    for e in graph["edges"]:
        G.add_edge(e["src"], e["dst"], edge_type=e["type"],
                  start_date=e["start_date"], end_date=e["end_date"], relation=e.get("relation", ""))

    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    plt.figure(figsize=(20, 16))

    node_config = {
        "person": ("lightblue", 'o', 2000),
        "location": ("lightgreen", 's', 1500),
        "event": ("orange", '^', 1500)
    }

    for ntype, (color, shape, size) in node_config.items():
        nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == ntype]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color,
                              node_shape=shape, node_size=size, alpha=0.9, label=ntype.title())

    edge_colors = {"person_to_person": "blue", "person_to_location": "green",
                   "person_to_event": "red", "event_to_location": "purple"}

    for etype, color in edge_colors.items():
        elist = [(u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == etype]
        if elist:
            nx.draw_networkx_edges(G, pos, edgelist=elist, edge_color=color,
                                  width=1.5, alpha=0.6, arrows=True, arrowsize=15)

    labels = {n: all_nodes[n]["name"][:15] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=6)

    plt.title("Temporal Graph", fontsize=16)
    plt.legend(loc='upper right')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def visualize_interactive(graph: dict, filename: str = "temporal_graph.html"):
    """Interactive pyvis visualization"""
    net = Network(height="900px", width="100%", bgcolor="#222222",
                  font_color="white", directed=True)
    net.set_options('{"physics": {"enabled": true, "barnesHut": {"gravitationalConstant": -8000}}}')

    node_colors = {"person": "#4ECDC4", "location": "#98D8C8", "event": "#FFA07A"}

    for nt in ["people", "locations", "events"]:
        for node in graph["nodes"][nt]:
            net.add_node(node["id"], label=node["name"][:20],
                        title=f"{node['type'].upper()}: {node['name']}",
                        color=node_colors[node["type"]],
                        size=30 if node["type"] == "person" else 20)

    edge_colors = {"person_to_person": "#0000FF", "person_to_location": "#00FF00",
                   "person_to_event": "#FF0000", "event_to_location": "#FF00FF"}

    for e in graph["edges"]:
        relation = e.get('relation', '')
        title = f"{e['type']}\n{e['start_date']} to {e['end_date']}"
        if relation:
            title += f"\n{relation}"
        net.add_edge(e["src"], e["dst"], title=title,
                    color=edge_colors[e["type"]], width=2, arrows='to')

    net.save_graph(filename)
    print(f"Saved: {filename}")

if __name__ == "__main__":
    print("=" * 80 + "\nTEMPORAL GRAPH GENERATION\n" + "=" * 80)

    graph = build_temporal_graph(num_people=100, num_locations=100, num_events=100,
                                  start_date="2024-01-01", num_days=30,
                                  nodes_per_day=10, edges_per_day=10, max_edge_duration_days=5000)

    print("\n" + "=" * 80 + "\nSAVING OUTPUTS\n" + "=" * 80)
    save_graph(graph)
    save_timeline(graph)
    visualize_matplotlib(graph)
    visualize_interactive(graph)

    print("\n" + "=" * 80 + "\nSUMMARY\n" + "=" * 80)
    print(f"Nodes: {len(graph['nodes']['people'])} people, "
          f"{len(graph['nodes']['locations'])} locations, "
          f"{len(graph['nodes']['events'])} events")
    print(f"Edges: {len(graph['edges'])} total, "
          f"{len(graph['edges']) / graph['metadata']['num_days']:.1f} avg/day")

    edge_types = {}
    for e in graph['edges']:
        edge_types[e['type']] = edge_types.get(e['type'], 0) + 1
    print("\nEdge breakdown:")
    for et, count in sorted(edge_types.items()):
        print(f"  {et}: {count}")

    print("\n" + "=" * 80)
    print("Done! Open temporal_graph.html to explore")
