"""
cluster.py
"""
import networkx as nx
import matplotlib.pyplot as plt
import pickle

    
   
    
def partition_girvan_newman(graph):

    G=graph.copy()
    
    if G.order() == 1:
        return [G.nodes()]
    
    eb =  nx.edge_betweenness_centrality(G) 
    #print(eb.items())
    edgelist= sorted(eb.items(), key=lambda x: (-x[1]))
    components=[c for c in nx.connected_component_subgraphs(G)]
     
        
    cnt=0
    while len(components) <= 4:
        edge_remove = edgelist[cnt] 
        G.remove_edge(edge_remove[0][0],edge_remove[0][1])
        components=[c for c in nx.connected_component_subgraphs(G)]
            
            
        #print(components)
        cnt+=1
        
      
    return components   
    pass
    


def get_subgraph(graph, min_degree):
 
    subgraphnodes=[]
    for node in graph.nodes():
       if graph.degree(node) >= min_degree:
          subgraphnodes.append(node)
    subgraph=graph.subgraph(subgraphnodes)
    return subgraph   
    pass


def draw_network(graph,filename):
    
    plt.figure(figsize=(12,12))
    labels={nodes: '' if isinstance(nodes,int) else nodes for nodes in graph.nodes()}
    nx.draw_networkx(graph,labels=labels, alpha=.5, width=.1,
                     node_size=100)
  
    plt.axis("off")
    plt.savefig(filename)
    pass
 
 
def main():
    graph =nx.read_gpickle("twittergraph.gpickle")
    print('graph has %d nodes and %d edges' % (graph.order(), graph.number_of_edges())) 
    #draw_network(graph,'network.png')      
    total=0
    #subgraph=get_subgraph(graph,1)
    #print('sub_graph has %d nodes and %d edges' % (subgraph.order(), subgraph.number_of_edges())) 
    clusters = partition_girvan_newman(graph)
    for i  in clusters:
        #print(type(i))
        total+=i.order()
    print('first partition: cluster 1 has %d nodes and cluster 2 has %d nodes' %
          (clusters[0].order(), clusters[1].order()))
    draw_network(clusters[0],"cluster-0.png")      
  
    with open('results.pkl', 'rb') as f:
        results = pickle.load(f)
    results['communities']= 2 
    results['average']=total/results['communities']
  
    draw_network(clusters[1],"cluster-1.png")       
    pickle.dump(results, open('results.pkl', 'wb'))
    print("Communities detected and graph drawn to cluster-0.png,cluster-1.png")
if __name__ == '__main__':
    main()  