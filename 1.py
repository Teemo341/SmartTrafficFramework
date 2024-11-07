import numpy as np
import pandas as pd
def decide_state(o_node,node1,node2,n_type):
        
        if n_type == 'Turning':
            return 0
        if n_type == 'T':
            if o_node[0]==2:
                if node2[1]>o_node[1]:
                    return 0
                else:
                    return 7
            if o_node[0]==1 and o_node[1]==1:
                if node2[0]==1:
                    return 5
                if node2[0]==3:
                    return 5
                if node2[0]==2 and node1[0]==1:
                    return 0
                if node2[0]==2 and node1[0]==2:
                    return 6
            if o_node[0]==1 and o_node[1]==10:
                if node2[0]==1:
                    return 5
                if node2[0]==3:
                    return 5
                if node2[0]==2 and node1[0]==1:
                    return 6
                if node2[0]==2 and node1[0]==2:
                    return 0
            if o_node[0]==10 and o_node[1]==1:
                if node2[0]==10:
                    return 5
                if node2[0]==8:
                    return 5
                if node2[0]==9 and node1[0]==10:
                    return 6
                if node2[0]==9 and node1[0]==9:
                    return 0
            if o_node[0]==10 and o_node[1]==10:
                if node2[0]==10:
                    return 5
                if node2[0]==8:
                    return 5
                if node2[0]==9 and node1[0]==10:
                    return 0
                if node2[0]==9 and node1[0]==9:
                    return 6
                
            if o_node[0]==1:
                if node2[0]==1:
                    return 5
                if node2[1]>o_node[1]:
                    return 0
                else:
                    return 6
            if o_node[0]==9:
                if node2[1]>o_node[1]:
                    return 7
                else:
                    return 0
            if o_node[0]==10:
                if node2[0]==10:
                    return 5
                if node2[1]>o_node[1]:
                    return 6
                else:
                    return 0
            if o_node[1]==2:
                if node2[0]>o_node[0]:
                    return 7
                else:
                    return 0
            if o_node[1]==1:
                if node2[1]==1:
                    return 5
                if node2[0]>o_node[0]:
                    return 6
                else:
                    return 0
            if o_node[1]==9:
                if node2[0]>o_node[0]:
                    return 0
                else:
                    return 7
            if o_node[1]==10:
                if node2[1]==10:
                    return 5
                if node2[0]>o_node[0]:
                    return 0
                else:
                    return 6
        if n_type == 'C':
            if o_node[0]-node2[0]==0:
                return 1
            if o_node[1]-node2[1]==0:
                return 3
            if o_node[0]-node1[0]==0: #南北
                if np.all(node2-o_node==[-1,1]):
                    return 2
                if np.all(node2-o_node==[1,1]):
                    return 0
                if np.all(node2-o_node==[-1,-1]):
                    return 2
                if np.all(node2-o_node==[1,-1]):
                    return 0
            if o_node[1]-node1[1]==0: #东西
                if np.all(node2-o_node==[1,1]):
                    return 4
                if np.all(node2-o_node==[1,-1]):
                    return 0
                if np.all(node2-o_node==[-1,1]):
                    return 4
                if np.all(node2-o_node==[-1,-1]):
                    return 0
                
import ast


state = []
node_edge = pd.read_csv('data/simulation/edge_node_10*10.csv')
node_type = pd.read_csv('data/simulation/node_type_10*10.csv')
node_edge1 = node_edge.copy()
node_edge1.rename(columns={'Origin':'Destination','Destination':'Origin'},inplace=True)
print(node_edge)
print(node_edge1)
node_edge = pd.concat([node_edge,node_edge1],axis=0,ignore_index=True)
results = pd.merge(node_edge,node_edge,left_on='Destination',right_on='Origin')
#results2 = pd.merge(node_edge,node_edge,left_on='Origin',right_on='Destination')
results = results.drop(columns=['Length_x','Length_y'])
#results2 = results2.drop(columns=['Length_x','Length_y'])
# results2 = results.copy()
# results2.rename(columns={'EdgeID_x':'EdgeID_y',
#                          'Origin_x':'Destination_y',
#                          'Destination_y':'Origin_x',
#                          'EdgeID_y':'EdgeID_x'},inplace=True)
# # print(results)
# print(results2)
#results = pd.concat([results, results2], axis=0, ignore_index=True)
o_node_list = []
node1_list = []
node2_list = []

for i in range(len(results)):
    o_node = node_type[node_type['NodeID']==results['Origin_x'][i]]['coordinate'].values[0]
    node1 = node_type[node_type['NodeID']==results['Origin_y'][i]]['coordinate'].values[0]
    node2 = node_type[node_type['NodeID']==results['Destination_y'][i]]['coordinate'].values[0]
    n_type = node_type[node_type['NodeID']==results['Destination_x'][i]]['Type'].values[0]
    o_node = ast.literal_eval(o_node)
    node1 =ast.literal_eval(node1)
    node2 =ast.literal_eval(node2)
    o_node=np.array(o_node)
    node1=np.array(node1)
    node2=np.array(node2)
    if decide_state(o_node,node1,node2,n_type) is None:
        print(o_node,node1,node2,n_type)
    state.append(decide_state(o_node,node1,node2,n_type))
    o_node_list.append(o_node)
    node1_list.append(node1)
    node2_list.append(node2)

results['state'] = state
results['o_node'] = o_node_list
results['node1'] = node1_list
results['node2'] = node2_list
# print(results)
results = results[results['EdgeID_x']!=results['EdgeID_y']]
results.to_csv('data/simulation/edge_node_10*10_state.csv',index=False)



