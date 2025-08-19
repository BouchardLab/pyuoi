#!/usr/bin/env python3

import numpy as np
import os
import argparse
import sys
import time
import networkx as nx
from toolbox.Util_NumpyIO import read_data_npz
from PlotterBetti import Plotter
from UtilBetti import print_betti_summary, BettiComputeres

from pprint import pprint

def extract_edges(bigD, md, args):
    """
    Convert 2D matrix A to NetworkX graph
    
    Args:
        bigD: data dictionary containing matrices
        md: metadata dictionary
        args: arguments with ampl_thres threshold
    
    Returns:
        nx.Graph: NetworkX graph with edges where abs(A[i,j]) > ampl_thres
    """
    if args.dataType == 'truth':  # truth input
        A = bigD['A_true']
    elif args.dataType == 'lasso':
        A = bigD['A_lasso'] 
    elif args.dataType == 'regress':
        A = bigD['A_pred']  # assuming regression results are in A_pred
    else:
        raise ValueError(f"Unknown dataType: {args.dataType}")
    
    print(f"Processing matrix A with shape: {A.shape}")
    print(f"Amplitude threshold: {args.ampl_thres}")
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add all nodes (assuming nodes are indexed 0 to n-1)
    n_nodes = A.shape[0]
    G.add_nodes_from(range(n_nodes))
    
    # Extract edges: reject diagonal, keep off-diagonal where abs(A[i,j]) > threshold
    edges_added = 0
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):  # Only upper triangular to avoid duplicates
            if abs(A[i, j]) > args.ampl_thres:
                # Add edge with weight as the matrix value
                G.add_edge(i, j, weight=A[i, j])
                edges_added += 1
    
    print(f"Matrix statistics:")
    print(f"  Diagonal elements: {n_nodes} (ignored)")
    print(f"  Off-diagonal elements: {n_nodes*(n_nodes-1)//2}")
    print(f"  Elements above threshold: {edges_added}")
    print(f"  edge fraction: {edges_added/(n_nodes*(n_nodes-1)//2)*100:.1f}%")
    
    return G
#########################
#  MAIN
#########################

def main():
    parser = argparse.ArgumentParser(description="Evaluate and plot results from fit_poisson.py")
    parser.add_argument("--dataName", type=str, default='dale_M120_3M', help="Base name for the dataset")
    parser.add_argument("--dataPath", type=str, default="out/", help="Path to the data directory")
    parser.add_argument('-a',"--ampl_thres", type=float, default=0.05, help="minima amplitude of valid off-diagonal edge")
    parser.add_argument('-k', '--betti_kmax', type=int, default=3,   help='Maximum dimension for Betti numbers')
    parser.add_argument('-m', '--maxItems', type=int, default=7, help='Maximum number of items to display (random subset for cycles/voids)')

    parser.add_argument('-p',"--showPlots", type=str,nargs='+', default=" ", help="Plot types to show")
    parser.add_argument("--outPath", type=str, default="out/", help="Output path for plots (defaults to dataPath)")
    parser.add_argument('-X',"--noXterm", action="store_true", help="Disable X terminal for plotting")
    parser.add_argument('-t',"--dataType", type=int, default=0, help="data type:  0=truth, 1:lasso, 2:regress")
    parser.add_argument('-v',"--verb", type=int, default=1, help="Verbosity level")
       
    args = parser.parse_args()
    
    if args.outPath is None:   args.outPath = args.dataPath
    typeD={0:'truth', 1:'lasso',2:'regress'}
    args.dataType=typeD[args.dataType]
    args.showPlots=''.join(args.showPlots)
    print(vars(args))

    assert args.betti_kmax>=1
    # Load graph data
    fitFF = os.path.join(args.dataPath, f"{args.dataName}.{args.dataType}.npz")

    if not os.path.exists(fitFF):
        print(f"Error: File not found: {fitFF}")
        return
            
    bigD, MD = read_data_npz(fitFF)
    if args.verb>1: 
        pprint(MD); exit(1)
    
    # Extract edges and create graph
    print(f"\n=== Extracting edges from {args.dataType} data ===")
    G = extract_edges(bigD, MD, args)
    
    # Print graph statistics
    print(f"\n=== Graph Statistics, input: {args.dataName} ===")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    
    if G.number_of_nodes() > 0:
        density = G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1) / 2)
        print(f"Graph density: {density:.4f}")
        
        # Check connectivity
        is_connected = nx.is_connected(G)
        print(f"Is connected: {is_connected}")
        
        if not is_connected:
            components = list(nx.connected_components(G))
            print(f"Number of connected components: {len(components)}")
            print(f"Largest component size: {len(max(components, key=len))}")
    
    # Store graph data for plotting if needed
    grfD = {'graph': G}
    grfMD = {'dataType': args.dataType, 'threshold': args.ampl_thres}

    # Compute Betti numbers
    print(f"\n=== Computing Betti numbers (max dimension: {args.betti_kmax}) ===")
    start_time = time.time()
    
    # Convert NetworkX graph to edge list for BettiComputer
    edges_list = np.array(list(G.edges), dtype=int)
    n_nodes = G.number_of_nodes()
    
    computer = BettiComputeres(max_dimension=args.betti_kmax)
    betti_numbers, simplex_counts, cycle_edges, void_triangles, void_tetrahedra = computer.compute(edges_list, n_nodes)
    
    computation_time = time.time() - start_time
    
    # Store results in grfD
    grfD['edges'] = edges_list
    grfD['cycle_edges'] = cycle_edges
    if args.betti_kmax >= 2:
        grfD['void_triangles'] = void_triangles
    if args.betti_kmax >= 3: 
        grfD['void_tetrahedra'] = void_tetrahedra

    grfMD = {
        'graph_conf': {
            'n_nodes': n_nodes,
            'n_edges': G.number_of_edges(),
            'max_dimension': args.betti_kmax,
            'edge_ampl_thres': args.ampl_thres,
            'data_type': args.dataType
        },
        'betti_results': {
            'betti_numbers': betti_numbers,
            'n_simplices': simplex_counts,
            'computation_time': computation_time
        },
        'short_name': f"betti_{args.dataName}_{args.dataType}_t{args.ampl_thres}"
    }
    
    print_betti_summary(grfD, grfMD, verbose=args.verb > 0)
    
    # Setup plotter if plots are requested
    if args.showPlots.strip():
        args.prjName = args.dataName 
        plot = Plotter(args)
        
        if 'a' in args.showPlots:
            plot.plot_cycles(grfD, grfMD, maxItems=args.maxItems, figId=1)
            
        if 'b' in args.showPlots:
            plot.plot_voids_2d(grfD, grfMD, maxItems=args.maxItems, figId=2)

        if 'c' in args.showPlots:
            plot.plot_voids_3d(grfD, grfMD, maxItems=args.maxItems, figId=3)

        plot.display_all()
    else:
        print("\nNo plots requested. Use -p flag to show plots (a=cycles, b=voids2D, c=voids3D)")
    

if __name__ == "__main__":
    main() 
