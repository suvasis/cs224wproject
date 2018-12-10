'''
Created on Nov 6, 2018

@author: suvmukhe Mukherjee and Minakshi Musherjee
'''
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import pandas as pd
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import time

import argparse
import numpy as np

from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor
from sklearn.metrics import r2_score
from sklearn.mixture import GMM
from sklearn import mixture
#from mpl_toolkits.mplot3d import Axes3D

from scipy.stats import pearsonr
import node2vec
from gensim.models import Word2Vec

#hierchical_clustering
def hierchical_clustering(G):
    E = list(G.edges())
    J = nx.jaccard_coefficient(G, E)
    clusters= {}
    cluster_number=-1

    threshold_value = 0.9
    total_number_of_iterations = 0

    while(True):
        total_number_of_iterations+=1
        print(total_number_of_iterations)

        all_nodes = list(G.nodes())
        final_s = -1
        final_t = -1
        print(total_number_of_iterations)
        parent_mini=9999999999999999999999
        # start of for
        for source in all_nodes:
            # print source
            lengths = nx.single_source_shortest_path_length(G, source)
            mini = 999999999999999999
            dest = -1
            for j in lengths:
                if lengths[j]<mini and lengths[j]!=0:
                    mini = lengths[j]
                    dest = j
            print("for",total_number_of_iterations)
            if mini<parent_mini:
                parent_mini=mini
                final_s = source
                final_t = dest
        print("outside for loop",total_number_of_iterations)
        # end of for        
        min_node = min(final_s, final_t)
        max_node = max(final_s, final_t)
        
        print( "min_node=", min_node)
        print( "max_node=", max_node)


        E = list(G.edges())
        
        # print "The min_node is %d", min_node
        # print "The max_node is %d", max_node        
        if min_node not in clusters and min_node not in  clusters:
            cluster_number+=1
            clusters[min_node]=cluster_number
            clusters[max_node]=cluster_number
        elif min_node not in clusters:
            clusters[min_node]=clusters[max_node]
        elif min_node not in  clusters:
            clusters[max_node]=clusters[min_node]
        else:
            if (max_node  in clusters ):
                cluster_max_node_belongs_to = clusters[max_node]
                clusters[max_node]=clusters[min_node]
                for u in clusters:
                    if u==cluster_max_node_belongs_to:
                        u=clusters[min_node]

        for u, v in E:
            if u==max_node and v!=min_node:
                G.add_edge(min_node, v)
            elif v==max_node and u!=min_node:
                G.add_edge(min_node, u)
        print("removing...")
        G.remove_node(max_node)
        E = list(G.edges())
        print("E list...")
        J = nx.jaccard_coefficient(G, E)
        

        value = 10 # set as infinity
        num=0
        for u, v, k in J:
            if k>threshold_value and k<value:
                value = k
                num+=1
        print ("num befor break",num)
        if num == 0:
            preds = J
            print("J is reaching... ")
            for u, v, p in J:
                print("J is reaching 1... ")
                print("jacard..(u, v) is a pair of nodes and p is their Jaccard coefficient..",u, v, p)
            break

    i=0
    unique_clusters = set()
    for u in clusters:
        # print 'The node ',i ,'belongs to the cluster number', u
        i+=1
        if u==-1:
            unique_clusters.add(i)
        else :
            unique_clusters.add(u)


    print( "The total number of unique clusters is", len(unique_clusters))
    print ("The total number of iterations", total_number_of_iterations)
    # networkx.draw(G,  with_labels=1)
    # plt.show()

#clustering
def printNetworkCharactics(G):
    n= nx.number_of_nodes(G)
    e = nx.number_of_edges(G)
    print(n,e)

    c= nx.clustering(G)
    d = nx.degree(G)
    print(c,d)
    return 


def localEfficiency(G):
    print("local efficiency",nx.local_efficiency(G))

    return 


def betweenness_centrality(G):
    print("betweennes centrality..",betweenness_centrality(G))
    return 

#jaccard index
def jaccardIndex(G, nodeid):
    n= nx.number_of_nodes(G)
    e = nx.number_of_edges(G)

    E = list(G.edges())
    print("E list...")
    J = nx.jaccard_coefficient(G, E)
    for u, v, k in J:
            if u==nodeid:
                print("jacard..(u, v) is a pair of nodes and p is their Jaccard coefficient..",u, v, k)

    return 
#pagerank
def get_pagerank(G):

        p = nx.pagerank_numpy(G, alpha=0.9)
        for n in G:
            print(p[n])
        personalize = dict((n, np.random.random()) for n in G)
        p = nx.pagerank_numpy(G, alpha=0.9, personalization=personalize)
        return p

def hits_nstart(G):

        nstart = dict([(i, 1. / 2) for i in G])
        h, a = nx.hits(G, nstart=nstart)
        print (h,a)
        
#clique community
def k_clique_communities(G, k): 
           
        #print("compoments")
        cliques=None
        if cliques is None:
            cliques = nx.find_cliques(G)
            print("compoments")
        cliques = [frozenset(c) for c in cliques if len(c) >= k]
        #print(">",cliques)
    
        # First index which nodes are in which cliques
        membership_dict = defaultdict(list)
        for clique in cliques:
            for node in clique:
                membership_dict[node].append(clique)
    
        # For each clique, see which adjacent cliques percolate
        perc_graph = nx.Graph()
        perc_graph.add_nodes_from(cliques)
        for clique in cliques:
            for adj_clique in _get_adjacent_cliques(clique, membership_dict):
                if len(clique.intersection(adj_clique)) >= (k - 1):
                    perc_graph.add_edge(clique, adj_clique)
    
        # Connected components of clique graph with perc edges
        # are the percolated cliques
        for component in nx.connected_components(perc_graph):
            
            yield(frozenset.union(*component))
    
        return

def _get_adjacent_cliques(clique, membership_dict):
    adjacent_cliques = set()
    for n in clique:
        for adj_clique in membership_dict[n]:
            if clique != adj_clique:
                adjacent_cliques.add(adj_clique)
    return adjacent_cliques

def getDataPointsToPlot(G):
    """
    :param - Graph: snap.PUNGraph object representing an undirected graph

    return values:
    X: list of degrees
    Y: list of frequencies: Y[i] = fraction of nodes with degree X[i]
    """
    ############################################################################
    # TODO: Your code here!
    his = nx.degree_histogram(G)
    
    X = []
    Y = []
    index=0
    for item in his:
        #print ("{} {}".format(item, index))
        X.append(index)
        Y.append(item)
        index +=1

    ############################################################################
    return X, Y

#node2vec learn embeddings
def learn_embeddings(walks):
        '''
        Learn embeddings by optimizing the Skipgram objective using SGD.
        '''
        walks = [map(str, walk) for walk in walks]
        model = Word2Vec(walks, size=128, window=10, min_count=0, sg=1, workers=8, iter=1)
        model.save_word2vec_format("../emb/milestone.emb")
        #model.wv.save_word2vec_format("../emb/milestone.emb")
 
        return model

#find node with largest degree
def nodeWithLargestDegree(G):

    #x_erdosRenyi, y_erdosRenyi = getDataPointsToPlot(erdosRenyi)
    #plt.loglog(x_erdosRenyi, y_erdosRenyi, color = 'y', label = 'Erdos Renyi Network')

    x_collabNet, y_collabNet = getDataPointsToPlot(G)
    plt.loglog(x_collabNet, y_collabNet, linestyle = 'dotted', color = 'b', label = 'Collaboration Network')

    plt.xlabel('Node Degree (log)')
    plt.ylabel('Proportion of Nodes with a Given Degree (log)')
    plt.title('Degree Distribution of Erdos Renyi, Small World, and Collaboration Networks')
    plt.legend()
    plt.show()
    
#normalize the vector
def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

#calculate score
def calculatescore(data,node,v, feature_vector):
    score = None
    mdf = data.loc[data['CONNID'] == int(node) ][feature_vector]

    a = None
    if len(mdf.index) != 0:
       a = np.array(mdf.values.tolist())
       va = normalize(a)
    
    s = []
    for nid in v:
        dfinner=data.loc[data['CONNID'] == int(nid)][feature_vector]
        b = None
        if len(dfinner.index) != 0:
           b = np.array(dfinner.values.tolist())
           vb = normalize(b)
        if a is not None and b is not None:
           s.append(va*vb)
    mat = np.matrix(np.array(s))
    colsum = mat.sum(axis=1)
    rowsum = colsum.sum(axis=0)
    return rowsum.item((0, 0))

#score
def getScores(data,dict,feature_vector):
    scores = []
    #print(data.shape)
    for k,v in dict.items():
        score = calculatescore(data,k,v,feature_vector)
        if score > 0:
           scores.append(score)
    return scores

#clean dataset - helper method
def clean_dataset(df):
    
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

# EM Gaussian mixture model
def em_gmm(data, scores):
    x = pd.DataFrame(scores)
    y = data.loc[:,'HDRS17_baseline']
    #y = data.loc[:,'SOFAS_baseline']
    #y = data.loc[:,'Nac_Clust2']#Insula_Clust2']#'Insula_Clus1'] #'Amygdala_Clust1']
    clean_dataset(x)

    #############
    # 2 components
    #################
    gmm = GMM(n_components=2).fit(x)
    labels = gmm.predict(x)
    ########################

    #####
    #plt.scatter(x.iloc[:, 0], y.iloc[:, 0], c=labels, s=40, cmap='viridis');
    #plt.scatter(x.iloc[:, 0], y, c=labels, s=40, cmap='viridis');
    plt.scatter(x.iloc[:, 0], y, c=labels, s=40, cmap='viridis')
    
    plt.xlabel('functional score')
    plt.ylabel('HDRS17_baseline');#'SOFAS_baseline'
    #plt.ylabel('Nac_Clust2')#'Insula_Clus2')#'Amygdala_Clust1')
    plt.title('Gaussian Mixture Model')
    #plt.colorbar();  # show color scale
    plt.show()

#pearson correlatio coeeficient
def corelcoeffpearsonr(data, scores):
    x = pd.DataFrame(scores)
    #y = data.loc[:,columns]#'HDRS17_baseline']
    y = data.loc[:,'HDRS17_baseline'] #'SOFAS_baseline'
    clean_dataset(x)
    #print(len(x),y.shape)
    correl = pearsonr(x.ravel(),y.ravel())
    return correl 

def linearRegression(data, scores):
    # use this dataset: - leave_one_out_results_linearregression.xlsx
    #data = pd.read_csv('C:/tools/workspace/cs224wproject/project/X_train.csv')#X_train.csv')#.head(150)
    #data = pd.read_excel('leave_one_out_results_linearregression.xlsx')#X_train.csv')#.head(150)
    data.columns = data.columns.to_series().apply(lambda x: x.strip())
    data=clean_dataset(data)
    data[data==np.inf]=np.nan
    data.fillna(data.mean(), inplace=True)
    columns =['CONNID', 'sertraline', 'venlafaxine', 'escitalopram', 'age', 'gender',
       'education', 'SOFAS_baseline', 'SOFAS_6',  'SOFAS_response', 'Amygdala_Clust1', 'Insula_Clus1', 'Insula_Clust2',
       'Nac_Clust1', 'Nac_Clust2']

    #x = data.loc[:, data.columns != 'HDRS17_baseline']
    #x = data.loc[:,columns]
    #print(x)
    x = pd.DataFrame(scores)
    y = data.loc[:,'HDRS17_baseline']
    linreg = LinearRegression()
    LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
    #print(x)
    linreg.fit(x,y)
    #print( linreg.predict(x))
    p = linreg.predict(x)
    err = abs(p-y)
    total_error = np.dot(err,err)
    rmse_train = np.sqrt(total_error/len(p))
    print (rmse_train)
    print ('Regression Coefficients: \n', linreg.coef_)
    plt.plot(p, y,'ro')
    plt.plot([0,50],[0,50], 'g-')
    plt.xlabel('predicted')
    plt.ylabel('real')
    plt.show()
    #print("test")

def driver():
    # use this dataset: - leave_one_out_results_linearregression.xlsx
    data = pd.read_excel('leave_one_out_results_linearregression.xlsx')#X_train.csv')#.head(150)

    column_ID = 'CONNID'
    feature_1 = 'sertraline'
    feature_2 = 'venlafaxine'
    feature_3 = 'escitalopram'
    feature_4 = 'age'
    feature_5 = 'gender'
    feature_6 = 'education'
    feature_7 = 'Amygdala_Clust1'
    feature_8 = 'Insula_Clus1'
    feature_9 = 'Insula_Clust2'
    feature_10 = 'Nac_Clust1'
    feature_11 = 'Nac_Clust2'


    
    feature_vector = [feature_1,feature_2,feature_3,feature_4,feature_5,feature_6,feature_7,feature_8,feature_9,feature_10,feature_11]
    data_to_merge = data[[column_ID, feature_vector]].dropna(subset=[feature_vector])#.dropna(subset=[column_edge1]).drop_duplicates() #select columns, remove NaN

    #create graph
    G = nx.Graph()
    nodes_for_adding=np.array(data.loc[:,'CONNID']) 
    #print(nodes_for_adding)
    #attaribute list
    attrlist = {}

    #creating nodes and adding attributes to the nodes.
    for nid in nodes_for_adding:
        
        df=data.loc[data['CONNID'] == nid]
        dict = {}
        if len(df.index) != 0:
            ff = df.to_dict()
            for key,val in ff.items():
                for k,v in val.items():  
                    dict[key] = v
                attrlist[nid] = dict
        #create node with attribute list
        G.add_node(nid, attr_dict=attrlist)


    #creating edges
    #algo
    k=15
    i=0
    n = len(nodes_for_adding)
    #print("n",n)
    #build the graph
    for i in range(1+k,n-k):
        k += 1
        i += 1
        for j in range(i-k,i+k):
            if i!=j and j<n:
               nid_i = nodes_for_adding[i]
               nid_j = nodes_for_adding[j]
               G.add_edge(nid_i,nid_j,weight=1)
               v_i = G.node[nid_i]
               v_j = G.node[nid_j]
            j += 1
##################################################################
    #create edges.
    edges = G.edges()
    #print("number of edges",len(edges))

    #####################node2vec################################
    # the value of p and q - to accurately access homophily
    ############################################################
    p = 10
    q = .1
    nv_G = node2vec.Graph(G, False, 100, .1)
    nv_G.preprocess_transition_probs()
    walks = nv_G.simulate_walks(5, 10)
    model = learn_embeddings(walks)
    #print("nodes ",G.nodes())
    l = list(G.nodes())
    walkmap = {}
    details = []
    for node in l:
        #print(" most similar : ",model.most_similar(str(node)))
        walkmap[node] = model.most_similar(str(node))
    for k,v in walkmap.items():
        for i in v:
            details.append(i[0])
        walkmap[k] = details
        #print("walkmap: ",k,walkmap[k])
        details = []

    dict = walkmap

    df = pd.DataFrame.from_dict(dict)
    scoresdict = {}
    for k,v in dict.items():
        #print(k,v)
        scoresdict[k] = v
    #print( scoresdict, len(scoresdict),data.shape)
    
    scores = getScores(data,scoresdict,feature_vector)
    #print ( scores)
    linearRegression(data,scores)
    em_gmm(data,scores)
    correl = corelcoeffpearsonr(data,scores)
    print(correl)
    featurelist = {}

    ###############################################################
    plt.figure(figsize=(20, 20))
    pos=nx.spring_layout(G, k=0.0319)
    nx.draw_networkx(G,pos,node_size=100, node_color='pink', font_size=8,
                        alpha=0.8)

    ##########################################################
    #Graph characteristics. 
    ###########################################################################

    printNetworkCharactics(G)
    # Compute the average clustering coefficient
    print(" avg clustering",nx.average_clustering(G))
    hierchical_clustering(G)
    jaccardIndex(G,140)
    localEfficiency(G)
    pos = nx.spring_layout(G)
    M=nx.authority_matrix(G,[163,140])
    #print(M)
    betweenness_centrality(G)
    pagerank = get_pagerank(G)
    for p in pagerank:
         print("pagerank for : ",p)

if __name__ == '__main__':
   driver()
