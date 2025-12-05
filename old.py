
import cohere
import requests
import random
import json
import time
from datetime import datetime
import names
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

"""
WORKFLOW: we want to create a graph of a family
1 family for now:

for each family:
- generate 3-5 family names (use a python module). Make sure no duplicate names in the entire graph!
- For each member, they have a relationship (edge) with EXACTLY 1 member of the family (randomly assigned) -- assign this edge (relationship)
- For each member, generate 10 random events in their life (with time, title, description). Each event becomes a node (its content is time/description/title) and create an edge between the member and each event: "had_event" edge.

"""

def retry(fn, *args, **kwargs):
    for i in range(10):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            safe_print(f"[{datetime.now()}] retry {i+1}/10 because: {e}")
            time.sleep(1 + i*0.3)
    raise RuntimeError("Failed after 10 retries")

def prompt_gpt(prompt):
    api_key = "..."
    url = "https://stg.api.cohere.ai/compatibility/v1/chat/completions"

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

def generate_random_relationship(name1, name2):
    def inner():
        PROMPT = f"""
Given names of these 2 people in the same family: {name1} and {name2}, generate the relationship of {name2} to {name1}.

Only provide the relationship of {name2} to {name1} under "Relationship:".
""".strip()
        relationship = prompt_gpt(PROMPT)
        parsed_name2_to_name1_relationship = relationship.split("Relationship:")[-1].strip()
        safe_print(f'Generated relationship of {name2} to {name1}: {parsed_name2_to_name1_relationship}')
        return relationship
    return retry(inner)


def generate_random_events(name):
    def inner():
        PROMPT = f"""
Given this person's name: {name}, I need you to generate 10 different events that happened in their life. For each event, provide the following in json format:
- time (in YYYY-MM-DD format)
- title (of the event)
- description (of the event)

Only provide the events in a list of json formats under "Events:".
""".strip()
        events = prompt_gpt(PROMPT)
        parsed_events = events.split("Events:")[-1].strip()
        parsed_events_evald = eval(parsed_events)
        safe_print(f'Generated events for {name}: {json.dumps(parsed_events_evald, indent=2)}')
        return parsed_events_evald
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


def build_family_graph(num_families=20, event_threads=10):
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
                futures[ex.submit(generate_random_events, m)] = m

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

        # relationships
        for m in members:
            safe_print(f"[{datetime.now()}] start relationship for {m}")
            possible_targets = [x for x in members if x != m]
            target = random.choice(possible_targets)
            rel = generate_random_relationship(target, m)
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
    return graph


def save_graph_to_file(graph, filename="family_graph.json"):
    with open(filename, 'w') as f:
        json.dump(graph, f, indent=2)
    print(f"Graph saved to {filename}")


def visualize_graph_matplotlib(graph, filename="family_graph.png"):
    G = nx.DiGraph()
    
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
                          edge_type="relationship",
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
    
    relationship_edges = [(u, v) for u, v, d in G.edges(data=True) 
                         if d.get("edge_type") == "relationship"]
    event_edges = [(u, v) for u, v, d in G.edges(data=True) 
                  if d.get("edge_type") == "had_event"]
    
    nx.draw_networkx_edges(G, pos, edgelist=relationship_edges, 
                          edge_color='blue', width=2, alpha=0.7,
                          arrows=True, arrowsize=20)
    nx.draw_networkx_edges(G, pos, edgelist=event_edges, 
                          edge_color='gray', width=1, alpha=0.3,
                          arrows=True, arrowsize=10)
    
    person_labels = {n: n for n in person_nodes}
    nx.draw_networkx_labels(G, pos, labels=person_labels, font_size=8)
    
    plt.title("Family Graph Visualization", fontsize=16)
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
                           title=rel_text,
                           color=color, width=3, arrows='to')
    
    net.save_graph(filename)
    print(f"Interactive visualization saved to {filename}")
    print(f"Open {filename} in your browser to explore the graph")


def test_generate_random_events():
    name = "Sungjin Hong"
    events = generate_random_events(name)
    print(f'Events for {name}: {json.dumps(events, indent=2)}')


def test_generate_random_relationship():
    name1 = "Sungjin Hong"
    name2 = "Varun Hong"
    relationship = generate_random_relationship(name1, name2)
    print(f'Relationship of {name2} to {name1}: {relationship}')



if __name__ == "__main__":
    print("Generating family graph...")
    # TODO (SJ): gen more families and somehow connect them
    graph = build_family_graph(1)  
    save_graph_to_file(graph, "family_graph.json")
    print("\nCreating static visualization...")
    visualize_graph_matplotlib(graph, "family_graph.png")
    breakpoint()
