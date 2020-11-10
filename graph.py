import networkx
import sys
import os
import json
import time


def run(filename=None, esc="\t"):
    g = networkx.DiGraph()
    f = open(filename)
    dict1=dict()
    dict2=dict()
    i=0
    for l in f: 
        a = l.replace("\x00","").strip().split(esc)
        i=i+1
        g.add_edge(int(a[0]), int(a[1]))##scratch batch graph
    #print(len(g.edges()),i)
    g1 = networkx.DiGraph()
    for p in range(10):##iterate the batches graph
        f = open('batches/batch_'+str(p)+'.txt')
        #i=0
        for l in f:
            a = l.strip().split(esc)
            #i=i+1
            #print(a)
            g1.add_edge(int(a[0]), int(a[1]))
        h=g1.copy()
        #print(len(g1.edges()),i)
        #i1=0

        for i,j in g.edges():
            try:
                #i1=i1+1
                #print(i,j)
                g1.remove_edge(i,j)
            except:
                 pass
        for i,j in h.edges():
            try:
                g.remove_edge(i,j)
            except:
                 pass
        #print(len(g1.edges()))
        #print(i1,"kk")
        networkx.write_edgelist(g, "batches/oldbatch_"+str(p)+".txt")
        networkx.write_edgelist(g1, "batches/newbatch_"+str(p)+".txt")
        g.add_edges_from(g1.edges()) 
        g1.clear()
        outll = dict()
        for n in g.nodes():
            for x in g.predecessors(n):
                dict1.setdefault(n,[]).append(x)
            for x in g.successors(n):
                dict2.setdefault(n,[]).append(x)
            outll[n] = len(g.out_edges(n))
            
        #print(g.number_of_edges())
        f1 = open("graph/Pred_"+str(p)+".txt","a")
        f2 = open("graph/Succ_"+str(p)+".txt","a")
        f3 = open("graph/Outll_"+str(p)+".txt","a")
        #f1.write( str(dict1) )
        json.dump(dict1, f1, indent=4)
        json.dump(dict2, f2, indent=4)
        json.dump(outll, f3, indent=4)
        #f1.write(str(dict2))
        f1.close()
        f2.close()
        dict1.clear()
        dict2.clear()
    



if __name__ == '__main__':
    os.system('rm batches/batch_* rm scratch_*')
    os.system('split --additional-suffix=".txt" -a 1 -l$[ $(wc -l {}|cut -d" " -f1) * {}/ 100 ] {} scratch_'.format(sys.argv[1],sys.argv[2],sys.argv[1]))
    os.system('split --additional-suffix=".txt" -a 1 -d -n l/{} ./scratch_b.txt batches/batch_'.format(sys.argv[3]))
    os.system('rm graph/Pred_* rm graph/Succ_* rm batches/newbatch_* rm batches/oldbatch_* rm graph/Outll_*')
    run(str("sc.txt"))

