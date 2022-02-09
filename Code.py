"""
Complex Networks final assignment
Pablo Rosillo
"""


import numpy as np
from tqdm import tqdm
import numba
import time
import csv
import graph_tool.all as gta
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import networkx.algorithms.community as nxc
import scipy.stats as scst
import scipy.linalg as scla
from scipy.optimize import curve_fit
import scipy.special as scsp
import pyperclip

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rc('font',**{'family':'serif','serif':['Times']})


# Read of adjacency list

Lpp = 4941
Nedges = 6594

Adj = np.zeros((Lpp, Lpp))

data = np.loadtxt("/Users/pablorosillo/OneDrive - Universitat de les Illes Balears/Máster IFISC/Asignaturas/Complex Networks/Final project/Power_grid.txt", usecols=(0,1), skiprows=0, dtype="int")

for i in range(len(data)):
    Adj[data[i][0]-1][data[i][1]-1] = 1
    
  
# Creation of graph in both networkx and graph_tools

g = gta.Graph()
gnx = nx.Graph()

for i in range(len(data)):
    g.add_edge(int(data[i][0]-1),int(data[i][1]-1), add_missing=True)
    gnx.add_edge(int(data[i][0]-1),int(data[i][1]-1))
g.set_directed(False)

gta.remove_parallel_edges(g)


# Katz centrality

ktzgraph = gta.katz(g, alpha = 0.01, beta = None)
ktz = [ktzgraph[i] for i in range(Lpp)]
ktz25i = np.argpartition(ktz,-25)[-25:]
ktz25 = np.partition(ktz,-25)[-25:]

# Degree assortativity

ass = gta.scalar_assortativity(g, deg="out")

# Watts-Strogatz coefficient

ws = gta.local_clustering(g)
wsvec = [ws[i] for i in range(Lpp)]
meanws = np.mean(wsvec)

# Subgraph centrality

sgc = nx.subgraph_centrality_exp(gnx)


# Best fit for the pdf

deg = g.get_out_degrees(g.get_vertices())
x = np.arange(Lpp)
h = plt.hist(deg, bins=range(Lpp))


accepted_dist_names = ['t', 'alpha', 'beta', 'betaprime', 'burr',
                       'foldnorm', 'genlogistic', 
                       'halfnorm', 'hypsecant', 'invgamma',
                       'invweibull', 'logistic',   'mielke']

h = plt.hist(deg, bins=range(Lpp))
for dist_name in tqdm(accepted_dist_names):
    dist = getattr(scst, dist_name)
    print(dist.name, dist.shapes)
    params = dist.fit(deg)
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    print(arg, loc, scale)
    if arg:
        pdf_fitted = dist.pdf(x, *arg, loc=loc, scale=scale)*Lpp
    else:
        pdf_fitted = dist.pdf(x, loc=loc, scale=scale)*Lpp
    plt.plot(pdf_fitted, label=dist_name)
    plt.xlim(0,47)
plt.legend(loc='upper right')
plt.show()



# Erdos-Renyi comparison (10 realizations)

def poissonf(x, C, lambd):
    y = C* np.exp(-lambd) * lambd**x /scsp.gamma(x+1) 
    return y

nrep = 10

density = Nedges*2/(Lpp*(Lpp-1))


WScoefferlist = []; Newmanerlist = []; avplerlist = []; diamerlist = []
degasserlist = []; biparterlist = []; degerlist = [];


for k in tqdm(range(nrep)):
    git = nx.erdos_renyi_graph(Lpp, density)
    if not nx.is_connected(git):
        ccl = sorted(nx.connected_components(git), key=len, reverse=True)
        git = git.subgraph(ccl[0])    
    WScoefferlist.append(nx.average_clustering(git))
    Newmanerlist.append(nx.transitivity(git))
    avplerlist.append(nx.average_shortest_path_length(git))
    diamerlist.append(nx.diameter(git))
    degasserlist.append(nx.degree_assortativity_coefficient(git))
    biparterlist.append(nx.bipartite.spectral_bipartivity(git))
    degerlist.append([git.degree[l] for l in git.nodes()])
    

meanWScoefer = np.mean(WScoefferlist)
stdWScoefer = np.std(WScoefferlist)
print(meanWScoefer, stdWScoefer)
meanNewmaner = np.mean(Newmanerlist)
stdNewmaner = np.std(Newmanerlist)
print(meanNewmaner, stdNewmaner)
meanavpler = np.mean(avplerlist)
stdavpler = np.std(avplerlist)
print(meanavpler, stdavpler)
meandiamer = np.mean(diamerlist)
stddiamer = np.std(diamerlist)
print(meandiamer, stddiamer)
meandegasser = np.mean(degasserlist)
stddegasser = np.std(degasserlist)
print(meandegasser, stddegasser)
meanbiparter = np.mean(biparterlist)
stdbiparter = np.std(biparterlist)
print(meanbiparter, stdbiparter)

degercounts = [];

fig = plt.figure(figsize=(3.5, 3))
bins = np.arange(0, 21, 1)
binsplot = bins[0:20]
for dist in degerlist:
    counts, binsa, bars = plt.hist(dist, bins=bins-0.5, density=True)
    degercounts.append(counts)

meandegercounts = np.mean(degercounts, axis=0)
stddegercounts = np.std(degercounts, axis=0)
plt.errorbar(x=binsplot, y=meandegercounts,yerr = stddegercounts/np.sqrt(nrep), fmt='ko', markersize=5)
plt.xlabel(r'$k$');  plt.ylabel('PDF')
plt.xticks([0,5,10,15,20])
fig.tight_layout()
# plt.savefig("er.eps")

parameterser, covarianceer = curve_fit(poissonf, binsplot, meandegercounts)
    
    
# Barabási-Albert comparison (10 realizations)

def powerlawf(x, C, lambd):
    y = C* x**lambd
    return y


WScoeffbalist = []; Newmanbalist = []; avplbalist = []; diambalist = []
degassbalist = []; bipartbalist = []; degbalist = [];
degbacounts = [];

for i in tqdm(range(nrep)):
    git = nx.barabasi_albert_graph(Lpp, m=2)
    if not nx.is_connected(git):
        ccl = sorted(nx.connected_components(git), key=len, reverse=True)
        git = git.subgraph(ccl[0])    
    WScoeffbalist.append(nx.average_clustering(git))
    Newmanbalist.append(nx.transitivity(git))
    avplbalist.append(nx.average_shortest_path_length(git))
    diambalist.append(nx.diameter(git))
    degassbalist.append(nx.degree_assortativity_coefficient(git))
    bipartbalist.append(nx.bipartite.spectral_bipartivity(git))
    degbalist.append([git.degree[l] for l in git.nodes()])
    

meanWScoefba = np.mean(WScoeffbalist)
stdWScoefba = np.std(WScoeffbalist)
print(meanWScoefba, stdWScoefba)
meanNewmanba = np.mean(Newmanbalist)
stdNewmanba = np.std(Newmanbalist)
print(meanNewmanba, stdNewmanba)
meanavplba = np.mean(avplbalist)
stdavplba = np.std(avplbalist)
print(meanavplba, stdavplba)
meandiamba = np.mean(diambalist)
stddiamba = np.std(diambalist)
print(meandiamba, stddiamba)
meandegassba = np.mean(degassbalist)
stddegassba = np.std(degassbalist)
print(meandegassba, stddegassba)
meanbipartba = np.mean(bipartbalist)
stdbipartba = np.std(bipartbalist)
print(meanbipartba, stdbipartba)

fig = plt.figure(figsize=(3.5, 3))
bins = np.arange(0, 21, 1)
binsplot = bins[0:20]
for dist in degbalist:
    counts, binsa, bars = plt.hist(dist, bins=bins-0.5, density=True)
    degbacounts.append(counts)

meandegbacounts = np.mean(degbacounts, axis=0)
stddegbacounts = np.std(degbacounts, axis=0)
plt.errorbar(x=binsplot, y=meandegbacounts,yerr = stddegbacounts/np.sqrt(nrep), fmt='ko', markersize=5)
plt.xlabel(r'$k$');  plt.ylabel('PDF')
plt.xticks([0,5,10,15,20])
#plt.savefig("ba.eps")


# SIS simulation

@numba.jit(nopython=True, parallel=True)
def simsis(T, betarray, mu, nststate, Adj):
    L = int(len(Adj))
    vertices = np.arange(0, L, 1)
    meanststate = [numba.int64(x) for x in range(0)]
    stdmeanststate = [numba.int64(x) for x in range(0)]
    randininf = np.random.randint(L)
    
    for beta in betarray:
        
        ststate = np.zeros(nststate, dtype=numba.int64)
        
        for m in numba.prange(nststate):
    
            htoday = np.zeros(L, dtype=numba.int64) # All susceptible
            htoday[randininf] = 1 # Random node infected
            
            I = 1;
    
            htomorrow = htoday
            
            for t in numba.prange(T):
                for u in vertices:
                    aux = np.delete(vertices, u)
                    if htoday[u] == 1:
                        for v in aux:
                            if Adj[u][v] == 1 and htoday[v] == 0:
                                if np.random.rand() < beta:
                                    htomorrow[v] = 1
                                    I += 1;
                        if np.random.rand() < mu:
                            htomorrow[u] = 0
                            I -= 1;
                htoday = htomorrow
                
            ststate[m] = I
            
        meanststate.append(np.mean(ststate))
        stdmeanststate.append(np.std(ststate))
        
    return meanststate, stdmeanststate


T = 500; nrealizations = 5*10**2;
betarray = np.linspace(0.001, 0.25, 250); mu = 0.1;

tic = time.time()
meaninf, stdmeaninf = simsis(T, betarray, mu, nrealizations, Adj)
toc = time.time()
print(toc-tic, 's elapsed')

errmeanststate = [aux/np.sqrt(nrealizations) for aux in stdmeaninf]

rows = zip(betarray, meaninf, errmeanststate)

name = "resultsnureddunaSIS"

f = open(name, "a+")
f.write(f"#Time elapsed: {toc-tic}\n")
f.write(f"#mu: {mu}\n#Beta_meanststate_errmeanststate\n")

writer = csv.writer(f)
for row in rows:
    writer.writerow(row)
    
f.close()


# Percolation: random errors and targeted attacks

betw, nothing = gta.betweenness(g) # Betweenness array
pgr = gta.pagerank(g) # PageRank array
clsn = gta.closeness(g) # Closeness array

nrep = 100; # Number of repetitions for random errors

# Vertex percolation

## Size of largest component

sgcr =np.zeros((nrep, Lpp));

verticesplot = sorted([v for v in g.vertices()], key=lambda v: betw[v])
sizes, comp = gta.vertex_percolation(g, verticesplot)
verticesplot = sorted([v for v in g.vertices()], key=lambda v: v.out_degree())
sizes3, comp3 = gta.vertex_percolation(g, verticesplot)
verticesplot = sorted([v for v in g.vertices()], key=lambda v: pgr[v])
sizes4, comp4 = gta.vertex_percolation(g, verticesplot)
verticesplot = sorted([v for v in g.vertices()], key=lambda v: clsn[v])
sizes5, comp5 = gta.vertex_percolation(g, verticesplot)

for i in tqdm(range(nrep)):

    
    np.random.shuffle(verticesplot)
    sizes2, comp2 = gta.vertex_percolation(g, verticesplot)
    
    sgcr[i] = sizes2


sizes2m = np.sum(sgcr, axis=0)/nrep
errsizes2m = np.std(sgcr, axis=0)/np.sqrt(nrep)

x = np.arange(0,Lpp,1)

plt.figure(2)
plt.plot(x/Lpp, sizes, '.', label=r"$k$-targeted attacks", markersize=2)
plt.plot(x/Lpp, sizes3,'.', label=r"Betweenness-targeted attacks", markersize=2)
plt.plot(x/Lpp, sizes5,'.', label=r"Closeness-targeted attacks", markersize=2)
plt.plot(x/Lpp, sizes4,'.', label=r"PageRank-targeted attacks", markersize=2)
plt.errorbar(x/Lpp, y=sizes2m, yerr=errsizes2m, fmt='.', label="Random errors", markersize=2)
plt.xlabel(r"$\phi_\mathrm{nodes}$")
plt.gca().invert_xaxis()
plt.ylabel("SLG")
plt.legend(loc="upper right")


## Number of connected components

betw, nothing = gta.betweenness(g) # Betweenness array
pgr = gta.pagerank(g) # PageRank array
clsn = gta.closeness(g) # Closeness array

# Degree-targeted

gaux = nx.Graph()

for i in tqdm(range(len(data)), desc="Graph creation"):
    gaux.add_edge(int(data[i][0]-1),int(data[i][1]-1))

vertices = np.arange(0, Lpp, 1)
verticesplot = sorted([v for v in vertices], key=lambda v: gnx.degree[v])

sizes = np.array([])

for v in tqdm(verticesplot, desc="Degree-targeted attack"):
    gaux.remove_node(v)
    sizes = np.append(sizes, nx.number_connected_components(gaux))
    
# Betweenness-targeted

gaux = nx.Graph()

for i in tqdm(range(len(data)), desc="Graph creation"):
    gaux.add_edge(int(data[i][0]-1),int(data[i][1]-1))


verticesplot = sorted([v for v in vertices], key=lambda v: betw[v])

sizes3 = np.array([])

for v in tqdm(verticesplot, desc="Betweenness-targeted attack"):
    gaux.remove_node(v)
    sizes3 = np.append(sizes3, nx.number_connected_components(gaux))
    
# Closeness-targeted

gaux = nx.Graph()

for i in tqdm(range(len(data)), desc="Graph creation"):
    gaux.add_edge(int(data[i][0]-1),int(data[i][1]-1))


verticesplot = sorted([v for v in g.vertices()], key=lambda v: clsn[v])

sizes5 = np.array([])

for v in tqdm(verticesplot, desc="Closeness-targeted attack"):
    gaux.remove_node(v)
    sizes5 = np.append(sizes5, nx.number_connected_components(gaux))
    
# PageRank-targeted

gaux = nx.Graph()

for i in tqdm(range(len(data)), desc="Graph creation"):
    gaux.add_edge(int(data[i][0]-1),int(data[i][1]-1))


verticesplot = sorted([v for v in g.vertices()], key=lambda v: pgr[v])

sizes4 = np.array([])

for v in tqdm(verticesplot, desc="PageRank-targeted attack"):
    gaux.remove_node(v)
    sizes4 = np.append(sizes4, nx.number_connected_components(gaux))
    
# Randomly targeted

nrep = 10
sgcr =np.zeros((nrep, Lpp));

for j in range(nrep):
    
    gaux = nx.Graph()

    for i in range(len(data)):
        gaux.add_edge(int(data[i][0]-1),int(data[i][1]-1))
    
    verticesplot = vertices
    np.random.shuffle(verticesplot)
    
    sizes2 = np.array([])
    
    for v in tqdm(verticesplot, desc=f"Random errors. Iteration {j+1} of {nrep}"):
        gaux.remove_node(v)
        sizes2 = np.append(sizes2, nx.number_connected_components(gaux))
    
    sgcr[j] = sizes2

sizes2m = np.sum(sgcr, axis=0)/nrep
errsizes2m = np.std(sgcr, axis=0)/np.sqrt(nrep)


x = np.arange(0,Lpp,1)
x = np.flip(x)

plt.figure(3)
plt.plot(x/Lpp, sizes, '.', label=r"$k$-targeted attacks", markersize=2)
plt.plot(x/Lpp, sizes3,'.', label=r"Betweenness-targeted attacks", markersize=2)
plt.plot(x/Lpp, sizes5,'.', label=r"Closeness-targeted attacks", markersize=2)
plt.plot(x/Lpp, sizes4,'.', label=r"PageRank-targeted attacks", markersize=2)
plt.errorbar(x/Lpp, y=sizes2m, yerr=errsizes2m, fmt='.', label="Random errors", markersize=2)
plt.xlabel(r"$\phi_\mathrm{nodes}$")
plt.gca().invert_xaxis()
plt.ylabel("NCC")
plt.legend(loc="upper left")




# Edge percolation

## Size of largest component

nrep = 100

sgcr =np.zeros((nrep, Nedges));

for i in tqdm(range(len(data)), desc="Graph creation"):
    gaux.add_edge(int(data[i][0]-1),int(data[i][1]-1))

edges = sorted([(e.source(), e.target()) for e in g.edges()],
           key=lambda e: e[0].out_degree() * e[1].out_degree())
sizes, comp = gta.edge_percolation(g, edges)

for i in tqdm(range(len(data)), desc="Graph creation"):
    gaux.add_edge(int(data[i][0]-1),int(data[i][1]-1))

edges = sorted([(e.source(), e.target()) for e in g.edges()],
           key=lambda e: (e[0].out_degree() + e[1].out_degree()))
sizes3, comp = gta.edge_percolation(g, edges)

for i in tqdm(range(len(data)), desc="Graph creation"):
    gaux.add_edge(int(data[i][0]-1),int(data[i][1]-1))

edges = sorted([(e.source(), e.target()) for e in g.edges()],
           key=lambda e: abs(e[0].out_degree() - e[1].out_degree()))
sizes4, comp = gta.edge_percolation(g, edges)


for i in tqdm(range(nrep)):

    np.random.shuffle(edges)
    sizes2, comp = gta.edge_percolation(g, edges)
    
    sgcr[i] = sizes2


sizes2m = np.sum(sgcr, axis=0)/nrep
errsizes2m = np.std(sgcr, axis=0)/np.sqrt(nrep)

x = np.arange(0,Nedges,1)

plt.figure(4)
plt.plot(x/Nedges, sizes, '.', label=r"$k_i k_j$-targeted attacks", markersize=2)
plt.plot(x/Nedges, sizes3, '.', label=r"$(k_i+k_j)$-targeted attacks", markersize=2)
plt.plot(x/Nedges, sizes4, '.', label=r"$|k_i-k_j|$-targeted attacks", markersize=2)
plt.errorbar(x/Nedges, y=sizes2m, yerr=errsizes2m, fmt='.', label="Random errors", markersize=2)
plt.xlabel(r"$\phi_\mathrm{edges}$")
plt.gca().invert_xaxis()
plt.ylabel("SLC")
plt.legend(loc="upper right")


## Number of connected components

nrep = 100

sgcr =np.zeros((nrep, Nedges));

gaux = nx.Graph()

for i in tqdm(range(len(data)), desc="Graph creation"):
    gaux.add_edge(int(data[i][0]-1),int(data[i][1]-1))

edges = sorted([(e.source(), e.target()) for e in g.edges()],
           key=lambda e: e[0].out_degree() * e[1].out_degree())

sizes = np.array([])

for e in tqdm(edges):
    gaux.remove_edge(e[0],e[1])
    sizes = np.append(sizes, nx.number_connected_components(gaux))
    
gaux = nx.Graph()
    
for i in tqdm(range(len(data)), desc="Graph creation"):
    gaux.add_edge(int(data[i][0]-1),int(data[i][1]-1))

edges = sorted([(e.source(), e.target()) for e in g.edges()],
           key=lambda e: (e[0].out_degree() + e[1].out_degree()))

sizes3 = np.array([])

for e in tqdm(edges):
    gaux.remove_edge(e[0],e[1])
    sizes3 = np.append(sizes3, nx.number_connected_components(gaux))
    
gaux = nx.Graph()
    
for i in tqdm(range(len(data)), desc="Graph creation"):
    gaux.add_edge(int(data[i][0]-1),int(data[i][1]-1))

edges = sorted([(e.source(), e.target()) for e in g.edges()],
           key=lambda e: abs(e[0].out_degree() - e[1].out_degree()))

sizes4 = np.array([])

for e in tqdm(edges):
    gaux.remove_edge(e[0],e[1])
    sizes4 = np.append(sizes4, nx.number_connected_components(gaux))


nrep = 10
sgcr =np.zeros((nrep, Nedges));

for j in range(nrep):
    
    gaux = nx.Graph()

    for i in range(len(data)):
        gaux.add_edge(int(data[i][0]-1),int(data[i][1]-1))
    
    edgesplot = edges
    np.random.shuffle(edgesplot)
    
    sizes2 = np.array([])
    
    for e in tqdm(edgesplot, desc=f"Random errors. Iteration {j+1} of {nrep}"):
        gaux.remove_edge(e[0],e[1])
        sizes2 = np.append(sizes2, nx.number_connected_components(gaux))
    
    sgcr[j] = sizes2

sizes2m = np.sum(sgcr, axis=0)/nrep
errsizes2m = np.std(sgcr, axis=0)/np.sqrt(nrep)

x = np.arange(0,Nedges,1)
x = np.flip(x)

plt.figure(5)
plt.plot(x/Nedges, sizes, '.', label=r"$k_i k_j$-targeted attacks", markersize=2)
plt.plot(x/Nedges, sizes3, '.', label=r"$(k_i+k_j)$-targeted attacks", markersize=2)
plt.plot(x/Nedges, sizes4, '.', label=r"$|k_i-k_j|$-targeted attacks", markersize=2)
plt.errorbar(x/Nedges, y=sizes2m, yerr=errsizes2m, fmt='.', label="Random errors", markersize=2)
plt.xlabel(r"$\phi_\mathrm{edges}$")
plt.gca().invert_xaxis()
plt.ylabel("NCC")
plt.legend(loc="upper left")



