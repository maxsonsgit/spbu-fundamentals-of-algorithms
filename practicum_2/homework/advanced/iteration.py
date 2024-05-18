import networkx as nx

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