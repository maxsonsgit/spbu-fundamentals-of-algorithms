import networkx as nx
# from iteration import iteration


def iteration(G: nx.DiGraph, vertex, visited: list[bool], flag: list[bool], start: list[str]):
    visited[int(vertex)] = True

    if not flag[0]:
        for neighbor in G.neighbors(vertex):
            if not (neighbor == start[0]):
                if not visited[int(neighbor)]:
                    iter(G, neighbor, visited, flag, start)
            else:
                flag[0] = True
                break
    return flag


def has_cycles(G: nx.DiGraph):
    ans = False
    for vertex in G:
        visited = [False] * len(G)
        if iteration(G, vertex, visited, [False], start=[vertex])[0]:
            ans = True
            break
    return ans



TEST_GRAPH_FILES = [
    "graph_1_wo_cycles.edgelist",
    "graph_2_wo_cycles.edgelist",
    "graph_3_wo_cycles.edgelist",
]

if __name__ == "__main__":
    for filename in TEST_GRAPH_FILES:
        # Load the graph
        G = nx.read_edgelist(filename, create_using=nx.DiGraph)

        ans = has_cycles(G)

        # Output whether it has cycles
        if ans:
            print(f"Graph {filename} has cycle")
        else:
            print(f"Graph {filename} has no cycle")
