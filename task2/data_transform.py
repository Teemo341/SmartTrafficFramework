import numpy as np
from lxml import etree
import csv
import scipy.sparse
from tqdm import tqdm

# note by zy:
# the 'shape' attribute is defined on 'lane' object in osm.net.xml, how to define the shape of 'edge'?
# here I simply access the shape of the first lane as edge's shape.
# note that the real edge id starts from 1 !!!

# this one is out of date
def xml2csv(input_file='data/osm.net.xml', output_file='data/edges_mapping.csv'):
    '''
    inputfile:
        osm.net.xml - roadnet file of sumo format

    outputfile:
        edges_mapping_withshape.csv - contains relationship between edge_id, shape and length
    '''
    tree = etree.parse(input_file)
    root = tree.getroot()
    
    edges = root.xpath('//edge')
    edge2attr = {}
    for edge in edges:
        id = edge.get('id')
        shape = edge.xpath('./lane/@shape')[0]
        lengths = [float(i) for i in edge.xpath('./lane/@length')]
        length = str(np.average(lengths))
        edge2attr[id] = {
            'shape': shape,
            'length': length
        }

    csv_file = output_file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Edge ID','Index','Shape', 'Length'])
        writer.writeheader()
        for i, (edgeid, attr) in enumerate(edge2attr.items()):
            writer.writerow({
                'Edge ID':edgeid, 
                'Index': i + 1, 
                'Shape': attr['shape'],
                'Length': attr['length']
                })

# 7.10
# use this funtion to generate data
def boston2adj(file_path='data/edges_mapping_node.csv', xml_path='data/osm.net.xml', 
               output_csv_file='data/edges_mapping.csv', output_adj_file='data/adjacent.npy'):
    tree = etree.parse('data/osm.net.xml')
    root = tree.getroot()

    # # Find the <location> element and extract the projParameter and netOffset attributes
    # location = root.find('location')
    # proj_params = location.get('projParameter')
    # net_offset = location.get('netOffset')
    # net_offset_x, net_offset_y = map(float, net_offset.split(','))

    # # Create the Proj objects for coordinate transformation
    # utm_proj = Proj(proj_params)
    # wgs84_proj = Proj(proj='latlong', datum='WGS84')
    # def to_geo(x, y):
    #     # Adjust by net offset
    #     adjusted_x = x - net_offset_x
    #     adjusted_y = y - net_offset_y
    #     # Convert to geographic coordinates
    #     lon, lat = transform(utm_proj, wgs84_proj, adjusted_x, adjusted_y)
    #     return lon, lat
    
    edges = tree.getroot().xpath("//edge[@id]")
    id2edges = {edge.get('id'):edge for edge in edges}
    u_nodes = []
    v_nodes = []
    lengths = []
    with open(file_path, 'r') as input_file, open(output_csv_file, 'w') as output_file:
        csvreader = csv.DictReader(input_file)
        csvwriter = csv.DictWriter(output_file, fieldnames=csvreader.fieldnames + ['from_loc', 'to_loc', 'shape'])
        csvwriter.writeheader()
        for i, row in enumerate(csvreader):
            from_node_id = int(row['from_node_id']) + 1
            to_node_id = int(row['to_node_id']) + 1
            length = float(row['length'])
            u_nodes.append(from_node_id)
            v_nodes.append(to_node_id)
            lengths.append(length)

            edge_id = row['edge_id']
            shape = id2edges[edge_id].xpath('./lane/@shape')[0]
            u, v = id2edges[edge_id].xpath('./@from')[0], id2edges[edge_id].xpath('./@to')[0]
            # note that the "" is necessary
            u = root.xpath(f'//junction[@id="{u}"]')[0]
            v = root.xpath(f'//junction[@id="{v}"]')[0]
            row['from_loc'] = u.get('x')+','+u.get('y')
            row['to_loc'] = v.get('x')+','+v.get('y')
            row['shape'] = shape
            row['from_node_id'] = str(from_node_id)
            row['to_node_id'] = str(to_node_id)
            
            csvwriter.writerow(row)
    # including unreal node 'zero'
    num_of_nodes = np.max((np.max(u_nodes), np.max(v_nodes)))

    adj_matrix = np.zeros((num_of_nodes + 1, num_of_nodes + 1))
    for u, v, length in zip(u_nodes, v_nodes, lengths):
        adj_matrix[u][v] = length
    np.save(output_adj_file, adj_matrix)

# 7.19
# the paris2adj not commited last time, now I have to rewrite it. 😭
def beijing2adj(file_path='/root/Porto/cropped_edges_info.csv', output_adj_file='/root/Porto/porto_3500'):
    u_nodes = []
    v_nodes = []
    lengths = []
    with open(file_path, 'r') as input_file:
        csvreader = csv.DictReader(input_file)
        for i, row in enumerate(csvreader):
            from_node_id = int(row['from_node_id'])
            to_node_id = int(row['to_node_id'])
            length = float(row['length'])
            u_nodes.append(from_node_id)
            v_nodes.append(to_node_id)
            lengths.append(length) 
            
    adj_matrix = scipy.sparse.csr_matrix((lengths, (u_nodes, v_nodes)))
    scipy.sparse.save_npz(output_adj_file, adj_matrix)

# 7.23
# zhouyi: this function is used to get a subset roadnet of Porto
def centercrop(file_path='Porto/edges_info.csv', out_path='Porto/cropped_edges_info.csv', 
               west=-8.649987, east=-8.612095, south=41.139844, north=41.175633):
    with open(file_path, 'r') as input_file, open(out_path, 'w') as output_file: 
        csvreader = csv.DictReader(input_file)
        csvwriter = csv.DictWriter(output_file, csvreader.fieldnames)
        csvwriter.writeheader()
        node_dic = {0: 0}

        def is_in_area(lon, lat):
            return west <= lon and lon <= east and south <= lat and lat <= north

        for row in csvreader:
            from_lon = float(row['from_lon'])
            from_lat = float(row['from_lat'])
            to_lon = float(row['to_lon'])
            to_lat = float(row['to_lat'])

            # filter by longitude and latitude
            if is_in_area(from_lon, from_lat) and is_in_area(to_lon, to_lat):
                from_node_id = int(row['from_node_id'])
                to_node_id = int(row['to_node_id'])

                # reassign node_id
                if from_node_id not in node_dic:
                    node_dic[from_node_id] = len(node_dic)
                if to_node_id not in node_dic:
                    node_dic[to_node_id] = len(node_dic)
                row['from_node_id'] = node_dic[from_node_id]
                row['to_node_id'] = node_dic[to_node_id]

                # write back to outputfile
                csvwriter.writerow(row)


if __name__ == "__main__":
    # boston2adj()
    # beijing2adj()
    centercrop()