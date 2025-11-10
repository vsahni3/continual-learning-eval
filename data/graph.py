import requests
import random
import json
import time
from datetime import datetime
import names
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pyvis.network import Network
import os
from dotenv import load_dotenv
load_dotenv()
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

"""
WORKFLOW: Family Graph Generation with Cross-Family Connections

This module generates synthetic family graphs with context-aware cross-family relationships
using LLM-based generation for realistic and consistent connections.

INTRA-FAMILY GENERATION:
For each family:
- Generate 3-5 unique family members with shared last name (no duplicate names across entire graph)
- Each member has a relationship edge with EXACTLY 1 other member (randomly assigned)
- Generate 10 life events per member (with time, title, description)
- Each event becomes a node with "had_event" edges to the member

CROSS-FAMILY CONNECTIONS:
- Connect multiple families with context-aware relationships
- LLM generates relationships considering ALL existing connections between families
- Ensures logical consistency (e.g., if A is B's cousin and C is A's sibling, then C should also be B's cousin)
- All connections generated with full relationship context
- Configurable density: 30-50% of people have cross-family connections
- Optional: Can keep some families disjoint

DATA STRUCTURE:
{
  "families": [
    {
      "members": [...],
      "edges": [...],  # intra-family relationships and events
      "events": [...]
    }
  ],
  "cross_family_edges": [  # NEW: relationships between families
    {
      "type": "relationship",
      "src": "Person1",
      "dst": "Person2",
      "relationship": "LLM-generated description",
      "src_family_idx": 0,
      "dst_family_idx": 1
    }
  ]
}

VISUALIZATION:
- Static (matplotlib): Blue edges for intra-family, green dashed for cross-family
- Interactive (pyvis): Family-colored nodes, bright green dashed edges for cross-family
"""

def retry(fn, *args, **kwargs):
    for i in range(10):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            safe_print(f"[{datetime.now()}] retry {i+1}/10 because: {e}")
            time.sleep(1 + i*0.3)
    raise RuntimeError("Failed after 10 retries")

def prompt_gpt(prompt, is_staging=False):
    api_key = os.getenv("COHERE_API_KEY")

    url = "https://stg.api.cohere.ai/compatibility/v1/chat/completions" if is_staging else "https://api.cohere.ai/compatibility/v1/chat/completions"

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"bearer {api_key}"
    }

    payload = {
        "model": "command-a-03-2025",
        # "reasoning_effort": "high",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=2000)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error during API request: {e}")
        raise e

    return content

def generate_relationship_from_prompt(prompt):
    def inner():
        relationship = prompt_gpt(prompt)
        parsed_relationship = relationship.split("Relationship:")[-1].strip().strip('**')
        return parsed_relationship
    return retry(inner)


def generate_intra_family_relationship(name1, name2, existing_relationships, family_members):
    """Generate relationship between two family members with context of existing relationships"""
    def inner():
        # Build context using existing relationships
        family_dict = {
            "members": family_members,
            "edges": existing_relationships
        }
        context = build_relationship_context([family_dict])

        PROMPT = f"""
{context}

GOAL: Generate high-quality, realistic synthetic family data. These are traditional families with 2 parents and n children/siblings.

Family structure:
- 2 parents (mother, father) who are spouses
- n children (son, daughter) who are siblings (brother, sister) to each other
- Not all relationships must be included, but those that ARE must be contradiction-free

CREATE a synthetic relationship: What is {name2} to {name1}? They are in the same family. 

Diversity is important, make sure not every relationship is simply 'brother' or 'sister'.

MUST pick from ONLY these exact types: "mother", "father", "son", "daughter", "brother", "sister", "spouse"
DO NOT use uncle, aunt, niece, nephew, cousin, or any other relationship type.

CRITICAL: NO CONTRADICTIONS with existing relationships - this must match a valid family tree structure.

Example: If A is brother of B, then B cannot be mother of A or spouse of A.

Stick to single word response, only provide the relationship under "Relationship:".
""".strip()
        relationship = prompt_gpt(PROMPT)
        parsed_relationship = relationship.split("Relationship:")[-1].strip().strip('**')
        safe_print(f'Generated relationship of {name2} to {name1}: {parsed_relationship}')
        print(PROMPT, parsed_relationship, '\n\n\n\n\n\n\n')
        return parsed_relationship
    return retry(inner)



def generate_random_events(name, num_events=1):
    def inner():
        PROMPT = f"""
Given this person's name: {name}, I need you to generate {num_events} different events that happened in their life. For each event, provide the following in json format:
- time (in YYYY-MM-DD format)
- title (of the event)
- description (of the event)

Only provide the events in a list of json formats under "Events:".
IMPORTANT: Do not use markdown code blocks. Do not wrap the response in ```json or ```. Return only the raw JSON array.
""".strip()
        events = prompt_gpt(PROMPT)
        parsed_events = events.split("Events:")[-1].strip()
        parsed_events_evald = json.loads(parsed_events)
        safe_print(f'Generated events for {name}: {json.dumps(parsed_events_evald, indent=2)}')
        return parsed_events_evald
    return retry(inner)


def build_relationship_context(families, existing_cross_relationships=None):
    """Build context string with all relationships - takes list of families"""
    context_parts = []

    for family in families:
        context_parts.append(f"Family {family['members'][0].split()[-1]} members and relationships:")
        context_parts.extend(f"  - {edge['src']} is {edge['relationship']} to {edge['dst']}" for edge in family["edges"] if edge["type"] == "relationship")

    if existing_cross_relationships:
        context_parts.append("\nExisting cross-family relationships:")
        context_parts.extend(f"  - {rel['src']} is {rel['relationship']} to {rel['dst']}" for rel in existing_cross_relationships)

    return "\n".join(context_parts)


def generate_cross_family_relationship(name1, name2, family1, family2, existing_cross_relationships):
    """Generate a single cross-family relationship with context of existing relationships"""
    def inner():
        context = build_relationship_context([family1, family2], existing_cross_relationships)

        PROMPT = f"""
{context}

GOAL: Generate high-quality, realistic synthetic cross-family relationship data with DIVERSE types. We need variety - if everything is cousin, that's bad data. Mix family relations with social connections (friends, coworkers) and activity-based relationships.

CREATE a synthetic relationship: What is {name2} to {name1}? They are from different families.

Relationship types:
- Family: cousin, second cousin, in-laws, uncle, aunt
- Social: childhood friend, college roommate, coworker, neighbor, business partner
- Activity: tennis partner, book club member, gym buddy

Pick to maximize diversity - include plenty of non-family types like friend, coworker, neighbor.

CRITICAL: Extended family relationships must be consistent. Example: if Theresa and Kevin are siblings, and Theresa is John's cousin, then Kevin must also be John's cousin.

FORMAT: Output ONLY the relationship type (1-3 words max). No explanations, no notes, no markdown.

Relationship:
""".strip()
        relationship = prompt_gpt(PROMPT)
        parsed_relationship = relationship.split("Relationship:")[-1].strip().strip('**')
        safe_print(f'Generated cross-family relationship of {name2} to {name1}: {parsed_relationship}')
        return parsed_relationship
    return retry(inner)



GLOBAL_USED_NAMES = set()
GLOBAL_USED_LAST_NAMES = set()

def generate_unique_last_name():
    """Generate a unique last name that hasn't been used before"""
    while True:
        last_name = names.get_last_name()
        if last_name not in GLOBAL_USED_LAST_NAMES:
            GLOBAL_USED_LAST_NAMES.add(last_name)
            return last_name

def generate_unique_names(n, last_name=None):
    """Generate n unique names, optionally with a shared last name"""
    out = []
    while len(out) < n:
        if last_name:
            first = names.get_first_name()
            full = f"{first} {last_name}"
        else:
            full = names.get_full_name()
        if full not in GLOBAL_USED_NAMES:
            GLOBAL_USED_NAMES.add(full)
            out.append(full)
    return out

print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


def add_cross_family_connections(graph, connection_ratio=0.4, max_family_pairs=None):
    """
    Add cross-family connections with context-aware LLM generation

    Args:
        graph: The family graph with families
        connection_ratio: Target ratio of people with cross-family connections (0.3-0.5)
        max_family_pairs: Maximum number of family pairs to connect (None = connect all)

    Returns:
        List of cross-family edges
    """
    families = graph["families"]
    num_families = len(families)
    cross_family_edges = []

    # Determine which family pairs to connect
    all_pairs = [(i, j) for i in range(num_families) for j in range(i+1, num_families)]

    # Optionally limit the number of pairs (to keep some families disjoint)
    pairs_to_connect = random.sample(all_pairs, max_family_pairs) if max_family_pairs and max_family_pairs < len(all_pairs) else all_pairs

    safe_print(f"[{datetime.now()}] Connecting {len(pairs_to_connect)} family pairs with context-aware relationships")

    for pair_idx, (i, j) in enumerate(pairs_to_connect):
        family1 = families[i]
        family2 = families[j]

        family1_name = family1["members"][0].split()[-1] 
        family2_name = family2["members"][0].split()[-1] 

        safe_print(f"[{datetime.now()}] Connecting families: {family1_name} <-> {family2_name} ({pair_idx+1}/{len(pairs_to_connect)})")

        # Track edges between this specific family pair
        edges_between_families = []

        # Calculate target number of connections using combinatorics
        # Max possible connections = family1_size * family2_size
        max_possible_connections = (len(family1["members"]) * len(family2["members"]))
        target_connections = max(1, int(max_possible_connections * connection_ratio))

        # Generate relationships one at a time with context
        # Sort pairs by sum of connections and go in order of minimum
        safe_print(f"[{datetime.now()}] Generating {target_connections} relationships one at a time with context...")

        # Track connection counts for each member
        connection_counts = {member: 0 for member in family1["members"] + family2["members"]}

        for _ in range(target_connections):
            # Create all possible pairs and sort by sum of connections
            all_pairs = [(m1, m2, connection_counts[m1] + connection_counts[m2])
                        for m1 in family1["members"]
                        for m2 in family2["members"]]
            all_pairs.sort(key=lambda x: x[2], reverse=True)  # Sort by sum of connections

            # Pick the pair with minimum connections
            member1, member2, _ = all_pairs.pop()

            safe_print(f"[{datetime.now()}] start cross-family relationship for {member1} -> {member2}")
            rel = generate_cross_family_relationship(member1, member2, family1, family2, edges_between_families)
            safe_print(f"[{datetime.now()}] finished cross-family relationship {member1} -> {member2}")

            edge = {
                "type": "relationship",
                "src": member1,
                "dst": member2,
                "relationship": rel,
                "src_family_idx": i,
                "dst_family_idx": j
            }
            edges_between_families.append(edge)
            cross_family_edges.append(edge)

            # Update connection counts
            connection_counts[member1] += 1
            connection_counts[member2] += 1

        safe_print(f"[{datetime.now()}] Completed {len(edges_between_families)} connections for {family1_name} <-> {family2_name}")

    safe_print(f"[{datetime.now()}] Total cross-family connections created: {len(cross_family_edges)}")
    return cross_family_edges


def build_family_graph(num_families=20, event_threads=10, num_events=1,
                       add_cross_family=True, connection_ratio=0.4,
                       max_family_pairs=None):
    """
    Build a family graph with optional cross-family connections

    Args:
        num_families: Number of families to generate
        event_threads: Number of threads for parallel event generation
        add_cross_family: Whether to add cross-family connections
        connection_ratio: Ratio of people with cross-family connections
        max_family_pairs: Maximum family pairs to connect (None = all pairs)

    Returns:
        Graph dictionary with families and optional cross_family_edges
    """
    graph = {
        "families": []
    }

    for f in range(num_families):
        safe_print(f"[{datetime.now()}] start family {f+1}/{num_families}")

        # Generate a unique last name for this family
        family_last_name = generate_unique_last_name()
        safe_print(f"[{datetime.now()}] family {f+1} last name: {family_last_name}")

        fam_size = random.randint(3,5)
        members = generate_unique_names(fam_size, last_name=family_last_name)

        # event parallel
        member_to_events = {}
        with ThreadPoolExecutor(max_workers=event_threads) as ex:
            futures = {}
            for m in members:
                safe_print(f"[{datetime.now()}] submitting events generation for member: {m}")
                futures[ex.submit(generate_random_events, m, num_events)] = m

            for fut in as_completed(futures):
                m = futures[fut]
                safe_print(f"[{datetime.now()}] finished events for member: {m}")
                ev_list = fut.result()
                member_to_events[m] = ev_list

        safe_print(f"[{datetime.now()}] ALL events done for family {f+1}")

        edges = []
        events_nodes = []
        # convert events → nodes
        for m, evs in member_to_events.items():
            for ev in evs:
                ev_node_id = f"{m}_{ev['time']}_{random.randint(0,999999)}"
                events_nodes.append({
                    "id": ev_node_id,
                    "type": "event",
                    "time": ev['time'],
                    "title": ev['title'],
                    "description": ev['description']
                })
                edges.append({
                    "type": "had_event",
                    "src": m,
                    "dst": ev_node_id
                })

        safe_print(f"[{datetime.now()}] generating relationships now for family {f+1}")

        for m in members:
            safe_print(f"[{datetime.now()}] start relationship for {m}")
            possible_targets = [x for x in members if x != m]
            target = random.choice(possible_targets)
            rel = generate_intra_family_relationship(target, m, edges, members)
            safe_print(f"[{datetime.now()}] finished relationship {m} -> {target}")
            edges.append({
                "type": "relationship",
                "src": m,
                "dst": target,
                "relationship": rel
            })

        safe_print(f"[{datetime.now()}] finished family {f+1}")

        graph["families"].append({
            "members": members,
            "edges": edges,
            "events": events_nodes
        })

    safe_print(f"[{datetime.now()}] all families done")

    # Add cross-family connections if requested
    if add_cross_family and num_families > 1:
        safe_print(f"[{datetime.now()}] starting cross-family connections...")
        cross_family_edges = add_cross_family_connections(
            graph,
            connection_ratio=connection_ratio,
            max_family_pairs=max_family_pairs,
        )
        graph["cross_family_edges"] = cross_family_edges
        safe_print(f"[{datetime.now()}] cross-family connections complete")
    else:
        graph["cross_family_edges"] = []

    return graph


def save_graph_to_file(graph, filename="family_graph.json"):
    with open(filename, 'w') as f:
        json.dump(graph, f, indent=2)
    print(f"Graph saved to {filename}")


def visualize_graph_matplotlib(graph, filename="family_graph.png"):
    G = nx.DiGraph()

    # Add intra-family nodes and edges
    for family_idx, family in enumerate(graph["families"]):
        for member in family["members"]:
            G.add_node(member, node_type="person", family=family_idx)

        # Add event nodes
        for event in family["events"]:
            G.add_node(event["id"], node_type="event",
                      title=event["title"], family=family_idx)

        for edge in family["edges"]:
            if edge["type"] == "had_event":
                G.add_edge(edge["src"], edge["dst"], edge_type="had_event")
            elif edge["type"] == "relationship":
                G.add_edge(edge["src"], edge["dst"],
                          edge_type="intra_family_relationship",
                          relationship=edge.get("relationship", ""))

    # Add cross-family edges
    if "cross_family_edges" in graph:
        for edge in graph["cross_family_edges"]:
            G.add_edge(edge["src"], edge["dst"],
                      edge_type="cross_family_relationship",
                      relationship=edge.get("relationship", ""))

    pos = nx.spring_layout(G, k=2, iterations=50)

    plt.figure(figsize=(20, 16))

    person_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "person"]
    event_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "event"]

    nx.draw_networkx_nodes(G, pos, nodelist=person_nodes,
                          node_color='lightblue', node_size=3000,
                          node_shape='o', alpha=0.9)

    nx.draw_networkx_nodes(G, pos, nodelist=event_nodes,
                          node_color='orange', node_size=500,
                          node_shape='s', alpha=0.6)

    # Separate edge types
    intra_family_edges = [(u, v) for u, v, d in G.edges(data=True)
                          if d.get("edge_type") == "intra_family_relationship"]
    cross_family_edges = [(u, v) for u, v, d in G.edges(data=True)
                          if d.get("edge_type") == "cross_family_relationship"]
    event_edges = [(u, v) for u, v, d in G.edges(data=True)
                  if d.get("edge_type") == "had_event"]

    # Draw intra-family relationships in blue
    nx.draw_networkx_edges(G, pos, edgelist=intra_family_edges,
                          edge_color='blue', width=2, alpha=0.7,
                          arrows=True, arrowsize=20)

    # Draw cross-family relationships in green
    nx.draw_networkx_edges(G, pos, edgelist=cross_family_edges,
                          edge_color='green', width=2.5, alpha=0.8,
                          arrows=True, arrowsize=20, style='dashed')

    # Draw event edges in gray
    nx.draw_networkx_edges(G, pos, edgelist=event_edges,
                          edge_color='gray', width=1, alpha=0.3,
                          arrows=True, arrowsize=10)

    person_labels = {n: n for n in person_nodes}
    nx.draw_networkx_labels(G, pos, labels=person_labels, font_size=8)

    # Add legend
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=2, label='Intra-family relationship'),
        Line2D([0], [0], color='green', linewidth=2.5, linestyle='dashed', label='Cross-family relationship'),
        Line2D([0], [0], color='gray', linewidth=1, label='Event connection')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.title("Family Graph Visualization (with Cross-Family Connections)", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Static visualization saved to {filename}")
    plt.close()


def visualize_graph_interactive(graph, filename="family_graph.html"):
    net = Network(height="900px", width="100%", bgcolor="#222222",
                  font_color="white", directed=True)

    # from claude sonnet:
    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "barnesHut": {
                "gravitationalConstant": -8000,
                "centralGravity": 0.3,
                "springLength": 200,
                "springConstant": 0.04
            },
            "minVelocity": 0.75
        }
    }
    """)

    # these colors are from claude sonnet
    family_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',
                     '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']

    for family_idx, family in enumerate(graph["families"]):
        color = family_colors[family_idx % len(family_colors)]

        for member in family["members"]:
            net.add_node(member,
                        label=member,
                        title=f"Family Member: {member}",
                        color=color,
                        size=30,
                        shape='dot',
                        font={'size': 20})

        for event in family["events"]:
            event_title = f"{event['title']}\n{event['time']}\n{event['description'][:50]}..."
            net.add_node(event["id"],
                        label="",
                        title=event_title,
                        color='#FFA500',
                        size=10,
                        shape='square')

        for edge in family["edges"]:
            if edge["type"] == "had_event":
                net.add_edge(edge["src"], edge["dst"],
                           color='#888888', width=1, arrows='to')
            elif edge["type"] == "relationship":
                rel_text = edge.get("relationship", "related to")
                net.add_edge(edge["src"], edge["dst"],
                           title=f"Intra-family: {rel_text}",
                           color=color, width=3, arrows='to')

    # Add cross-family edges in bright green
    if "cross_family_edges" in graph:
        for edge in graph["cross_family_edges"]:
            rel_text = edge.get("relationship", "connected to")
            net.add_edge(edge["src"], edge["dst"],
                        title=f"Cross-family: {rel_text}",
                        color='#00FF00',  # Bright green
                        width=4,
                        arrows='to',
                        dashes=True)  # Dashed line to distinguish

    net.save_graph(filename)
    print(f"Interactive visualization saved to {filename}")
    print(f"Open {filename} in your browser to explore the graph")


def test_generate_random_events():
    name = "Sungjin Hong"
    events = generate_random_events(name)
    print(f'Events for {name}: {json.dumps(events, indent=2)}')





if __name__ == "__main__":
    print("Generating family graph with cross-family connections...")

    # Configuration for cross-family connections
    NUM_FAMILIES = 10  # Generate 3 families
    CONNECTION_RATIO = 0.4  # % of people get cross-family connections
    MAX_FAMILY_PAIRS = 20

    graph = build_family_graph(
        num_families=NUM_FAMILIES,
        event_threads=10,
        num_events=10,
        add_cross_family=True,
        connection_ratio=CONNECTION_RATIO,
        max_family_pairs=MAX_FAMILY_PAIRS
    )

    save_graph_to_file(graph, "family_graph.json")

    print("\nCreating static visualization...")
    visualize_graph_matplotlib(graph, "family_graph.png")

    print("\nCreating interactive visualization...")
    visualize_graph_interactive(graph, "family_graph.html")

    print("\n=== Graph Summary ===")
    print(f"Total families: {len(graph['families'])}")
    print(f"Total cross-family connections: {len(graph.get('cross_family_edges', []))}")

    for idx, family in enumerate(graph["families"]):
        family_name = family["members"][0].split()[-1] if family["members"] else f"Family{idx}"
        print(f"  {family_name}: {len(family['members'])} members")

    print("\nVisualization files created:")
    print("  - family_graph.json (data)")
    print("  - family_graph.png (static visualization)")
    print("  - family_graph.html (interactive visualization)")

    breakpoint()