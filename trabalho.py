import os
import json
import webbrowser
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pyvis.network import Network

# Configurações visuais dos gráficos
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

# ==============================================================================
# 1. CARREGAMENTO, LIMPEZA E CONSTRUÇÃO DO GRAFO REAL
# ==============================================================================
ARQUIVO_CSV = "transfers.csv"
df = pd.read_csv(ARQUIVO_CSV)

# Converter data e filtrar recorte recente para manter o grafo gerenciável (> 300 nós)
df["transfer_date"] = pd.to_datetime(df["transfer_date"], errors="coerce")
df = df[df["transfer_date"] >= "2026-01-01"].copy()

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

# ==============================================================================
# 5. PLOT 1: DISTRIBUIÇÃO DE 4 MÉTRICAS DO GRAFO REAL
# ==============================================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(f"Distribuição de Métricas - Grafo Transfermarkt (N={N}, M={M})", fontsize=14, fontweight="bold")

sns.histplot(df_nos["Grau"], bins=30, kde=True, ax=axes[0, 0], color="#1f77b4")
axes[0, 0].set_title("Distribuição de Grau (Número de Conexões)")
axes[0, 0].set_xlabel("Grau (k)")
axes[0, 0].set_ylabel("Quantidade de Vértices")

sns.histplot(df_nos["Clustering"], bins=25, kde=True, ax=axes[0, 1], color="#2ca02c")
axes[0, 1].set_title("Distribuição do Coeficiente de Agrupamento (Clustering)")
axes[0, 1].set_xlabel("Coeficiente de Clustering")
axes[0, 1].set_ylabel("Quantidade de Vértices")

sns.histplot(df_nos["Betweenness"], bins=30, kde=False, ax=axes[1, 0], color="#d62728")
axes[1, 0].set_title("Distribuição da Centralidade de Intermediação (Betweenness)")
axes[1, 0].set_xlabel("Betweenness Centrality")
axes[1, 0].set_ylabel("Quantidade de Vértices (escala log)")
axes[1, 0].set_yscale("log")

sns.histplot(df_nos["Closeness"], bins=25, kde=True, ax=axes[1, 1], color="#9467bd")
axes[1, 1].set_title("Distribuição da Centralidade de Proximidade (Closeness)")
axes[1, 1].set_xlabel("Closeness Centrality")
axes[1, 1].set_ylabel("Quantidade de Vértices")

plt.tight_layout()
plt.savefig("01_distribuicao_metricas_real.png", bbox_inches="tight")
plt.close()

# ==============================================================================
# 6. PLOT 2: COMPARAÇÃO COM OS GRAFOS SINTÉTICOS (LOG-LOG + BARRAS)
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Comparação: Grafo Real vs. Modelos Sintéticos Equivalentes", fontsize=14, fontweight="bold")

cores = {
    "Real (Transfermarkt)": "#e63946",
    "Aleatório (ER)": "#457b9d",
    "Small-World (WS)": "#2a9d8f",
    "Livre de Escala (BA)": "#f4a261",
}

# 6A) Distribuição de Grau Acumulada Complementar (CCDF) em escala Log-Log
for nome, G in grafos.items():
    graus = np.sort([d for _, d in G.degree() if d > 0])
    valores, contagens = np.unique(graus, return_counts=True)
    ccdf = 1.0 - np.cumsum(contagens) / len(graus) + (contagens / len(graus))
    axes[0].loglog(valores, ccdf, marker="o", markersize=4, linestyle="-", label=nome, color=cores[nome])

axes[0].set_title("Distribuição de Grau (CCDF Log-Log)")
axes[0].set_xlabel("Grau k (log)")
axes[0].set_ylabel("P(Grau ≥ k) (log)")
axes[0].legend()

# 6B) Clustering Médio vs Caminho Médio
sns.barplot(x=tabela_comp.index, y=tabela_comp["Clustering Médio"], hue=tabela_comp.index, palette=cores, legend=False, ax=axes[1])
axes[1].set_title("Coeficiente de Agrupamento Médio")
axes[1].set_xlabel("")
axes[1].set_ylabel("Clustering Médio")
axes[1].tick_params(axis="x", rotation=20)

sns.barplot(x=tabela_comp.index, y=tabela_comp["Caminho Médio (LCC)"], hue=tabela_comp.index, palette=cores, legend=False, ax=axes[2])
axes[2].set_title("Menor Caminho Médio (LCC)")
axes[2].set_xlabel("")
axes[2].set_ylabel("Distância Média")
axes[2].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig("02_comparacao_sinteticos.png", bbox_inches="tight")
plt.close()

# ==============================================================================
# 7. GRAFOS DINÂMICOS INTERATIVOS (GERAL + SEPARADO POR COMUNIDADES)
# ==============================================================================
comunidades = nx.community.louvain_communities(G_real, weight="weight", seed=42)
mapa_comunidade = {}
for idx, comm in enumerate(comunidades):
    for node in comm:
        mapa_comunidade[node] = idx

print("\n=== RESUMO DAS COMUNIDADES (CORES) ===")
for idx, comm in enumerate(comunidades):
    top_clubes = sorted(comm, key=lambda n: grau_dict[n], reverse=True)[:5]
    print(f"Comunidade {idx} ({len(comm)} clubes): {', '.join(top_clubes)}")


def salvar_grafo_pyvis(pos_dict, escala, nome_arquivo, abrir=True):
    net = Network(
        height="850px",
        width="100%",
        bgcolor="#111827",
        font_color="#f9fafb",
        select_menu=True,
        filter_menu=False,
        cdn_resources="remote",
    )

    limite_label = np.percentile(list(grau_dict.values()), 85)

    for node in G_real.nodes():
        grau = int(grau_dict[node])
        btw = float(betweenness_dict[node])
        pr = float(pagerank_dict[node])
        comm = int(mapa_comunidade[node])
        x, y = pos_dict[node]

        tooltip = (
            f"Clube: {node}\n"
            f"Grau (clubes conectados): {grau}\n"
            f"Betweenness: {btw:.4f}\n"
            f"PageRank: {pr:.4f}\n"
            f"Comunidade: {comm}"
        )

        net.add_node(
            str(node),
            label=str(node) if grau >= limite_label else "",
            title=tooltip,
            size=float(8 + (grau * 0.7)),
            group=comm,
            x=float(x * escala),
            y=float(y * escala),
        )

    for u, v, data in G_real.edges(data=True):
        w = int(data.get("weight", 1))
        net.add_edge(
            str(u),
            str(v),
            value=w,
            title=f"{u} ↔ {v} ({w} transferências)",
            color={
                "color": "rgba(156, 163, 175, 0.18)",
                "highlight": "#facc15",  # Amarelo ao clicar
                "hover": "#38bdf8",      # Azul ciano ao passar o mouse
            },
        )

    opcoes = {
        "nodes": {
            "font": {"size": 14, "strokeWidth": 2, "strokeColor": "#111827"},
            "borderWidth": 1.5,
            "borderWidthSelected": 3,
        },
        "edges": {
            "smooth": False,
            "scaling": {"min": 1, "max": 5},
        },
        "interaction": {
            "hover": True,
            "hoverConnectedEdges": True,
            "selectConnectedEdges": True,
            "tooltipDelay": 80,
            "hideEdgesOnDrag": True,
            "hideEdgesOnZoom": True,
            "navigationButtons": True,
        },
        "physics": {"enabled": False},
    }

    net.set_options(json.dumps(opcoes))

    html = net.generate_html()
    with open(nome_arquivo, "w", encoding="utf-8") as f:
        f.write(html)

    caminho = os.path.abspath(nome_arquivo)
    print(f"Gerado: {caminho}")
    if abrir:
        webbrowser.open(f"file://{caminho}")


# --- HTML 1: Layout Geral ---
print("\nGerando HTML 1 (Layout Geral)...")
pos_geral = nx.spring_layout(G_real, k=1.8 / np.sqrt(N), iterations=80, seed=42)
salvar_grafo_pyvis(pos_geral, escala=2200, nome_arquivo="grafo_01_geral.html", abrir=True)

# --- HTML 2: Layout Separado por Comunidades (Ilhas) ---
print("Gerando HTML 2 (Layout por Comunidades)...")
pos_grupos = {}
num_comms = len(comunidades)
raio_global = 2200

for idx, comm in enumerate(comunidades):
    subgrafo = G_real.subgraph(comm)
    angulo = 2 * np.pi * idx / num_comms
    centro_x = raio_global * np.cos(angulo)
    centro_y = raio_global * np.sin(angulo)

    sub_pos = nx.spring_layout(subgrafo, k=1.5 / np.sqrt(len(comm)), iterations=60, seed=42)
    escala_local = 350 + len(comm) * 6

    for node, (x, y) in sub_pos.items():
        pos_grupos[node] = (centro_x + x * escala_local, centro_y + y * escala_local)

salvar_grafo_pyvis(pos_grupos, escala=1, nome_arquivo="grafo_02_comunidades.html", abrir=True)