import pandas as pd
import numpy as np
import networkx as nx

# ==============================================================================
# 1. CARREGAMENTO, LIMPEZA E CONSTRUÇÃO DO GRAFO REAL
# ==============================================================================
ARQUIVO_CSV = "transfers.csv"
df = pd.read_csv(ARQUIVO_CSV)

# Converter data e filtrar recorte recente para manter o grafo gerenciável (> 300 nós)
df["transfer_date"] = pd.to_datetime(df["transfer_date"], errors="coerce")
df = df[df["transfer_date"] >= "2022-01-01"].copy()

# Remover entidades inválidas e nulos
invalidos = {"Without Club", "Retired", "Unknown", "Career break", "Ban"}
df = df.dropna(subset=["from_club_name", "to_club_name"])
df = df[
    (~df["from_club_name"].isin(invalidos)) &
    (~df["to_club_name"].isin(invalidos)) &
    (df["from_club_name"] != df["to_club_name"])
]

# Agregar arestas (peso = número de transferências entre dois clubes)
arestas = (
    df.groupby(["from_club_name", "to_club_name"])
    .size()
    .reset_index(name="weight")
)

# Grafo não-direcionado para comparação direta com ER, WS e BA
G_full = nx.Graph()
for row in arestas.itertuples(index=False):
    u, v, w = row.from_club_name, row.to_club_name, row.weight
    if G_full.has_edge(u, v):
        G_full[u][v]["weight"] += w
    else:
        G_full.add_edge(u, v, weight=w)

# Extrair Maior Componente Conexo (LCC)
lcc_nodes = max(nx.connected_components(G_full), key=len)
G_real = G_full.subgraph(lcc_nodes).copy()

# Se ainda estiver grande demais para visualização (> 700 nós), aplica k-core mantendo >= 300 nós
if G_real.number_of_nodes() > 700:
    for k in [2, 3, 4]:
        G_temp = nx.k_core(G_real, k=k)
        if G_temp.number_of_nodes() >= 300:
            lcc_temp = max(nx.connected_components(G_temp), key=len)
            if len(lcc_temp) >= 300:
                G_real = G_temp.subgraph(lcc_temp).copy()
        else:
            break

N = G_real.number_of_nodes()
M = G_real.number_of_edges()
k_medio = 2 * M / N
print(f"Grafo Real -> Vértices (N): {N} | Arestas (M): {M} | Grau médio: {k_medio:.2f}")

# ==============================================================================
# 2. GERAÇÃO DOS GRAFOS SINTÉTICOS EQUIVALENTES
# ==============================================================================
# 1) Aleatório (Erdos-Renyi) - mesmo N e mesmo M
G_er = nx.gnm_random_graph(n=N, m=M, seed=42)

# 2) Small-World (Watts-Strogatz) - k deve ser par e próximo do grau médio
k_ws = max(2, int(round(k_medio)))
if k_ws % 2 != 0:
    k_ws += 1
G_ws = nx.watts_strogatz_graph(n=N, k=k_ws, p=0.1, seed=42)

# 3) Livre de Escala (Barabasi-Albert) - m = arestas por novo nó (~ M/N)
m_ba = max(1, int(round(M / N)))
G_ba = nx.barabasi_albert_graph(n=N, m=m_ba, seed=42)

grafos = {
    "Real (Transfermarkt)": G_real,
    "Aleatório (ER)": G_er,
    "Small-World (WS)": G_ws,
    "Livre de Escala (BA)": G_ba,
}

# ==============================================================================
# 3. CÁLCULO DAS MÉTRICAS (REAL E SINTÉTICOS)
# ==============================================================================
def calcular_metricas_globais(G):
    lcc = G.subgraph(max(nx.connected_components(G), key=len))
    graus = [d for _, d in G.degree()]
    return {
        "Vértices": G.number_of_nodes(),
        "Arestas": G.number_of_edges(),
        "Grau Médio": np.mean(graus),
        "Grau Máximo": np.max(graus),
        "Clustering Médio": nx.average_clustering(G),
        "Caminho Médio (LCC)": nx.average_shortest_path_length(lcc),
        "Diâmetro (LCC)": nx.diameter(lcc),
        "Densidade": nx.density(G),
        "Assortatividade": nx.degree_assortativity_coefficient(G),
    }

tabela_comp = pd.DataFrame(
    {nome: calcular_metricas_globais(g) for nome, g in grafos.items()}
).T
print("\n=== COMPARAÇÃO DE MÉTRICAS GLOBAIS ===")
print(tabela_comp.round(4))
tabela_comp.to_csv("comparacao_metricas.csv")

# Métricas por vértice no grafo real
grau_dict = dict(G_real.degree())
clustering_dict = nx.clustering(G_real)
betweenness_dict = nx.betweenness_centrality(G_real)
closeness_dict = nx.closeness_centrality(G_real)
pagerank_dict = nx.pagerank(G_real, weight="weight")

df_nos = pd.DataFrame({
    "Clube": list(G_real.nodes()),
    "Grau": [grau_dict[n] for n in G_real.nodes()],
    "Clustering": [clustering_dict[n] for n in G_real.nodes()],
    "Betweenness": [betweenness_dict[n] for n in G_real.nodes()],
    "Closeness": [closeness_dict[n] for n in G_real.nodes()],
    "PageRank": [pagerank_dict[n] for n in G_real.nodes()],
})

# ==============================================================================
# 4. TOP 5 VÉRTICES MAIS IMPORTANTES POR CENTRALIDADE
# ==============================================================================
print("\n=== TOP 5 CLUBES POR MÉTRICA DE CENTRALIDADE ===")
for col in ["Grau", "Betweenness", "Closeness", "PageRank"]:
    print(f"\nTop 5 por {col}:")
    print(df_nos.sort_values(by=col, ascending=False)[["Clube", col]].head(5).to_string(index=False))