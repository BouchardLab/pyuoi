#!/usr/bin/env python3

import numpy as np
import gudhi
import argparse
import time
from typing import Dict, Tuple, List
from dataclasses import dataclass
import json
import networkx as nx
from PlotterBetti import Plotter

@dataclass
class BettiResults:
    """Container for Betti number computation results"""
    n_nodes: int
    edge_probability: float
    n_edges: int
    betti_numbers: Dict[int, int]
    computation_time: float
    n_simplices: Dict[int, int]
    cycle_edges: List[List[Tuple[int, int]]] = None  # List of cycles, each cycle is a list of edges
    
class RandomGraphGenerator:
    """CPU-based random graph generator"""
    
    def __init__(self, n_nodes: int, edge_prob: float, seed: int = None):
        self.n_nodes = n_nodes
        self.edge_prob = edge_prob
        if seed is not None:
            np.random.seed(seed)
    
    def generate(self) -> Tuple[np.ndarray, int]:
        """
        Generate random graph using CPU
        Returns: (edges_array, n_edges)
        """
        # Generate upper triangular part only (no self-loops)
        n_possible = self.n_nodes * (self.n_nodes - 1) // 2
        
        # Generate random values for all possible edges
        random_vals = np.random.random(n_possible)
        edge_mask = random_vals < self.edge_prob
        
        # Convert to edge list
        edges = self._mask_to_edges(edge_mask)
        n_edges = len(edges)
        
        return edges, n_edges
    
    def _mask_to_edges(self, edge_mask: np.ndarray) -> np.ndarray:
        """Convert boolean mask to edge list"""
        edge_indices = np.where(edge_mask)[0]
        n_edges = len(edge_indices)
        
        if n_edges == 0:
            return np.empty((0, 2), dtype=np.int32)
        
        edges = np.empty((n_edges, 2), dtype=np.int32)
        
        # Convert linear indices to (i,j) pairs
        # Using inverse of k = i*n - i*(i+1)/2 + j - i - 1
        for idx, k in enumerate(edge_indices):
            k = int(k)
            # Solve for i: i = floor((2*n - 1 - sqrt((2*n-1)^2 - 8*k))/2)
            i = int((2*self.n_nodes - 1 - np.sqrt((2*self.n_nodes-1)**2 - 8*k)) / 2)
            j = k - i*self.n_nodes + i*(i+1)//2 + i + 1
            edges[idx, 0] = i
            edges[idx, 1] = j
        
        return edges

class BettiComputer:
    """Compute Betti numbers using GUDHI"""
    
    def __init__(self, max_dimension: int = 3):
        self.max_dimension = max_dimension
    
    def compute(self, edges: np.ndarray, n_nodes: int) -> Tuple[Dict[int, int], Dict[int, int], List[List[Tuple[int, int]]], List[List[List[int]]], List[List[List[int]]]]:
        """
        Compute Betti numbers up to max_dimension
        Returns: (betti_numbers, simplex_counts, cycle_edges, void_triangles, void_tetrahedra)
        """
        # Build simplex tree
        st = gudhi.SimplexTree()
        
        # Add vertices
        for v in range(n_nodes):
            st.insert([v])
        
        # Add edges
        for edge in edges:
            st.insert(edge.tolist())
        
        # CRITICAL FIX: Only expand if we actually want higher dimensional features
        # For dense graphs, expansion creates too many simplices
        if self.max_dimension > 1:
            # Use flag complex for clique detection, but limit dimension
            # This is more memory efficient than full expansion
            print(f"   Building flag complex up to dimension {self.max_dimension}...")
            st.expansion(self.max_dimension + 1)
        
        # Count simplices by dimension
        simplex_counts = {dim: 0 for dim in range(self.max_dimension + 2)}
        for simplex, _ in st.get_filtration():
            dim = len(simplex) - 1
            if dim <= self.max_dimension + 1:
                simplex_counts[dim] += 1
        
        print(f"   Computing homology...")
        # Compute persistence and Betti numbers
        st.compute_persistence()
        
        # CRITICAL FIX: Get persistence intervals to compute Betti numbers (compatible with this GUDHI version)
        betti_numbers = {}
        
        # Method 1: Direct computation for graphs (most reliable for β₀ and β₁)
        if self.max_dimension >= 0:
            # β₀ = number of connected components
            # Can be computed from 0-dimensional persistence
            intervals0 = st.persistence_intervals_in_dimension(0)
            betti_numbers[0] = int(np.sum(np.isinf(intervals0[:, 1]))) if intervals0.size > 0 else 0
            if betti_numbers[0] == 0:  # Fallback
                betti_numbers[0] = 1  # Assume connected if edges exist
        
        if self.max_dimension >= 1:
            intervals1 = st.persistence_intervals_in_dimension(1)
            betti_numbers[1] = int(np.sum(np.isinf(intervals1[:, 1]))) if intervals1.size > 0 else 0
        
        # For higher dimensions, use GUDHI's computation
        for dim in range(2, self.max_dimension + 1):
            intervals_d = st.persistence_intervals_in_dimension(dim)
            betti_numbers[dim] = int(np.sum(np.isinf(intervals_d[:, 1]))) if intervals_d.size > 0 else 0
        
        # Sanity check and warning
        # Extract cycles from 1D persistence and voids from 2D and 3D persistence
        cycle_edges = []
        void_triangles = []
        void_tetrahedra = []
        if self.max_dimension >= 1 and betti_numbers[1] > 0:
            # Get persistence pairs for dimension 1
            intervals1 = st.persistence_intervals_in_dimension(1)
            persistent_cycles = []
            
            if intervals1.size > 0:
                # Count infinite intervals (persistent cycles)
                infinite_mask = np.isinf(intervals1[:, 1])
                n_persistent = int(np.sum(infinite_mask))
                
                # For visualization, create representative cycles from the graph structure
                # Using a systematic approach to find independent cycles
                G = nx.Graph()
                G.add_edges_from(edges)
                
                # Find a spanning tree
                if G.number_of_nodes() > 0 and nx.is_connected(G):
                    spanning_tree = nx.minimum_spanning_tree(G)
                    tree_edges = set(spanning_tree.edges())
                    
                    # Non-tree edges create fundamental cycles
                    cycle_count = 0
                    for edge in edges:
                        if cycle_count >= n_persistent:
                            break
                        u, v = edge[0], edge[1]
                        if (u, v) not in tree_edges and (v, u) not in tree_edges:
                            # Find path in spanning tree between u and v
                            try:
                                tree_path = nx.shortest_path(spanning_tree, u, v)
                                # Create cycle: tree_path + direct edge back
                                cycle_edge_list = []
                                for i in range(len(tree_path) - 1):
                                    cycle_edge_list.append((tree_path[i], tree_path[i+1]))
                                cycle_edge_list.append((v, u))  # Close the cycle
                                cycle_edges.append(cycle_edge_list)
                                cycle_count += 1
                            except:
                                continue
                else:
                    # For disconnected graphs, use simple cycle detection
                    all_cycles = []
                    for component in nx.connected_components(G):
                        subG = G.subgraph(component)
                        if subG.number_of_edges() > subG.number_of_nodes() - 1:
                            # This component has cycles
                            try:
                                cycles = nx.minimum_cycle_basis(subG)
                                all_cycles.extend(cycles)
                            except:
                                continue
                    
                    # Convert to edge format
                    for i, cycle in enumerate(all_cycles):
                        if i >= n_persistent:
                            break
                        cycle_edge_list = []
                        for j in range(len(cycle)):
                            u, v = cycle[j], cycle[(j+1) % len(cycle)]
                            cycle_edge_list.append((u, v))
                        cycle_edges.append(cycle_edge_list)
        
        # Extract 2D voids from persistence
        if self.max_dimension >= 2 and betti_numbers[2] > 0:
            intervals2 = st.persistence_intervals_in_dimension(2)
            if intervals2.size > 0:
                # Count infinite intervals (persistent 2D voids)
                infinite_mask = np.isinf(intervals2[:, 1])
                n_voids = int(np.sum(infinite_mask))
                
                # Find triangles that bound the voids
                # Get all 2-simplices (triangles) from the filtration
                triangles_in_complex = []
                for simplex, _ in st.get_filtration():
                    if len(simplex) == 3:  # Triangle
                        triangles_in_complex.append(sorted(simplex))
                
                # For visualization, select representative triangles for each void
                # This is a simplified approach - in practice, void boundaries are more complex
                if triangles_in_complex and n_voids > 0:
                    # Group triangles by connectivity to identify void boundaries
                    triangle_groups = []
                    used_triangles = set()
                    
                    for i, triangle in enumerate(triangles_in_complex):
                        if i in used_triangles or len(triangle_groups) >= n_voids:
                            continue
                        
                        # Start a new group with this triangle
                        current_group = [triangle]
                        used_triangles.add(i)
                        
                        # Find connected triangles (sharing an edge)
                        for j, other_triangle in enumerate(triangles_in_complex):
                            if j in used_triangles:
                                continue
                            
                            # Check if triangles share an edge
                            shared_edges = 0
                            for k in range(3):
                                edge1 = (triangle[k], triangle[(k+1)%3])
                                for l in range(3):
                                    edge2 = (other_triangle[l], other_triangle[(l+1)%3])
                                    if (edge1[0] == edge2[0] and edge1[1] == edge2[1]) or \
                                       (edge1[0] == edge2[1] and edge1[1] == edge2[0]):
                                        shared_edges += 1
                            
                            if shared_edges > 0 and len(current_group) < 6:  # Limit size for visualization
                                current_group.append(other_triangle)
                                used_triangles.add(j)
                        
                        triangle_groups.append(current_group)
                    
                    void_triangles = triangle_groups[:n_voids]
        
        # Extract 3D voids from persistence
        if self.max_dimension >= 3 and betti_numbers[3] > 0:
            intervals3 = st.persistence_intervals_in_dimension(3)
            if intervals3.size > 0:
                # Count infinite intervals (persistent 3D voids)
                infinite_mask = np.isinf(intervals3[:, 1])
                n_3d_voids = int(np.sum(infinite_mask))
                
                # Find tetrahedra that bound the 3D voids
                # Get all 3-simplices (tetrahedra) from the filtration
                tetrahedra_in_complex = []
                for simplex, _ in st.get_filtration():
                    if len(simplex) == 4:  # Tetrahedron
                        tetrahedra_in_complex.append(sorted(simplex))
                
                # For visualization, select representative tetrahedra for each 3D void
                if tetrahedra_in_complex and n_3d_voids > 0:
                    # Group tetrahedra by connectivity to identify 3D void boundaries
                    tetrahedra_groups = []
                    used_tetrahedra = set()
                    
                    for i, tetrahedron in enumerate(tetrahedra_in_complex):
                        if i in used_tetrahedra or len(tetrahedra_groups) >= n_3d_voids:
                            continue
                        
                        # Start a new group with this tetrahedron
                        current_group = [tetrahedron]
                        used_tetrahedra.add(i)
                        
                        # Find connected tetrahedra (sharing a triangular face)
                        for j, other_tetrahedron in enumerate(tetrahedra_in_complex):
                            if j in used_tetrahedra:
                                continue
                            
                            # Check if tetrahedra share a triangular face
                            shared_faces = 0
                            for k in range(4):
                                # Get triangular face of first tetrahedron
                                face1 = sorted([tetrahedron[l] for l in range(4) if l != k])
                                for m in range(4):
                                    # Get triangular face of second tetrahedron
                                    face2 = sorted([other_tetrahedron[l] for l in range(4) if l != m])
                                    if face1 == face2:
                                        shared_faces += 1
                            
                            if shared_faces > 0 and len(current_group) < 4:  # Limit size for visualization
                                current_group.append(other_tetrahedron)
                                used_tetrahedra.add(j)
                        
                        tetrahedra_groups.append(current_group)
                    
                    void_tetrahedra = tetrahedra_groups[:n_3d_voids]
        
        if self.max_dimension >= 2 and simplex_counts.get(3, 0) > 100000:
            print(f"   ⚠️  Warning: {simplex_counts[3]:,} tetrahedra found - this may affect accuracy")
        
        return betti_numbers, simplex_counts, cycle_edges, void_triangles, void_tetrahedra

    def compute_simple(self, edges: np.ndarray, n_nodes: int) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Alternative: Compute only β₀ and β₁ without expansion (fast and accurate)
        """
        st = gudhi.SimplexTree()
        
        # Add all simplices
        for v in range(n_nodes):
            st.insert([v])
        for edge in edges:
            st.insert(edge.tolist())
        
        # No expansion - just work with graph
        st.compute_persistence()
        
        # Count components using persistence intervals (compatible API)
        intervals0 = st.persistence_intervals_in_dimension(0)
        betti_0 = int(np.sum(np.isinf(intervals0[:, 1]))) if intervals0.size > 0 else 0
        if betti_0 == 0:
            betti_0 = 1
        
        # Use Euler characteristic for cycles
        betti_1 = len(edges) - n_nodes + betti_0
        
        return {0: betti_0, 1: betti_1}, {0: n_nodes, 1: len(edges)}

def print_summary(grfD: dict, grfMD: dict, verbose: bool = True):
    """Print formatted summary of results"""
    
    print("\n" + "="*60)
    print("RANDOM GRAPH BETTI NUMBER COMPUTATION SUMMARY")
    print("="*60)
    
    # Extract data from the structured dictionaries
    graph_conf = grfMD['graph_conf']
    betti_results = grfMD['betti_results']
    
    # Graph properties
    print(f"\n📊 Graph Properties:")
    print(f"   Nodes: {graph_conf['n_nodes']}")
    print(f"   Edge probability: {graph_conf['edge_probability']:.1%}")
    print(f"   Actual edges: {graph_conf['n_edges']:,}")
    print(f"   Expected edges: {int(graph_conf['n_nodes']*(graph_conf['n_nodes']-1)/2 * graph_conf['edge_probability']):,}")
    print(f"   Average degree: {2*graph_conf['n_edges']/graph_conf['n_nodes']:.1f}")
    
    # Simplex counts
    if verbose and betti_results.get('n_simplices'):
        print(f"\n🔺 Simplex Counts:")
        simplex_names = {0: "vertices", 1: "edges", 2: "triangles", 
                        3: "tetrahedra", 4: "4-simplices"}
        for dim, count in betti_results['n_simplices'].items():
            if count > 0:  # Only show non-zero counts
                name = simplex_names.get(dim, f"{dim}-simplices")
                print(f"   {name:12}: {count:,}")
    
    # Betti numbers
    print(f"\n🎯 Betti Numbers:")
    interpretations = {
        0: "connected components",
        1: "independent cycles",
        2: "voids/cavities",
        3: "3D voids"
    }
    
    betti_numbers = betti_results['betti_numbers']
    for dim in range(4):
        value = betti_numbers.get(dim, 0)
        interpretation = interpretations.get(dim, f"{dim}-dimensional holes")
        symbol = f"β_{dim}"
        print(f"   {symbol} = {value:4d}  ({interpretation})")
    

    # Performance
    print(f"\n⏱️  Computation Time: {betti_results['computation_time']:.3f} seconds")
    

def save_results(results: BettiResults, filename: str):
    """Save results to JSON file"""
    data = {
        'n_nodes': results.n_nodes,
        'edge_probability': results.edge_probability,
        'n_edges': results.n_edges,
        'betti_numbers': results.betti_numbers,
        'computation_time': results.computation_time,
        'n_simplices': results.n_simplices
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {filename}")

def main():
    parser = argparse.ArgumentParser(
        description='Compute Betti numbers for random graphs'
    )
    parser.add_argument('-n', '--nodes', type=int, default=100,
                       help='Number of nodes in the graph (default: 500)')
    parser.add_argument('-g', '--probability', type=float, default=0.15,
                       help='Edge probability (default: 0.15)')
    parser.add_argument('-d', '--max-dim', type=int, default=3,
                       help='Maximum dimension for Betti numbers (default: 3)')
    parser.add_argument('-s', '--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('-v', '--verb', type=int, default=1,
                       help='Verbosity level')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output JSON file for results')

    parser.add_argument('--outPath', type=str, default='out/',
                       help='Output path for plots')
    parser.add_argument('-X', '--noXterm', action='store_true',
                       help='Disable X terminal for plotting')
    parser.add_argument('-p',"--showPlots", type=str,nargs='+', default=" ", help="Plot types to show: a=cycles, b=voids2D, c=voids3D")

    args = parser.parse_args()
    args.showPlots=''.join(args.showPlots) 
    print(f"\n🚀 Starting Betti number computation...")
    print(f"   Configuration: {args.nodes} nodes, {args.probability:.1%} edge probability")
    
    start_time = time.time()
    
    # Generate random graph
    generator = RandomGraphGenerator(args.nodes, args.probability, args.seed)
    g_edges, n_edges = generator.generate()
    
    # Compute Betti numbers
    computer = BettiComputer(max_dimension=args.max_dim)
    
  
    betti_numbers, simplex_counts, cycle_edges, void_triangles, void_tetrahedra = computer.compute(g_edges, args.nodes)
    
    computation_time = time.time() - start_time
    
    # Prepare graph data structures following the pattern from sim_dale_poissonV5.py
    grfD = {
        'edges': g_edges,
        'cycle_edges': cycle_edges if ('a' in args.showPlots or args.verb > 1) else [],
        'void_triangles': void_triangles if ('b' in args.showPlots or args.verb > 1) else [],
        'void_tetrahedra': void_tetrahedra if ('c' in args.showPlots or args.verb > 1) else []
    }
    
    grfMD = {
        'graph_conf': {
            'n_nodes': args.nodes,
            'edge_probability': args.probability,
            'n_edges': n_edges,
            'seed': args.seed,
            'max_dimension': args.max_dim,
        },
        'betti_results': {
            'betti_numbers': betti_numbers,
            'n_simplices': simplex_counts if args.verb > 1 else {},
            'computation_time': computation_time
        },
        'short_name': f"betti_{args.nodes}n_{int(args.probability*100)}p"
    }
    
    # Store results (keep for backwards compatibility)
    results = BettiResults(
        n_nodes=args.nodes,
        edge_probability=args.probability,
        n_edges=n_edges,
        betti_numbers=betti_numbers,
        computation_time=computation_time,
        n_simplices=simplex_counts if args.verb > 1 else {},
        cycle_edges=cycle_edges if args.verb > 1 else []
    )
    
    # Print summary
    print_summary(grfD, grfMD, verbose=args.verb > 0)
    
    # Save results if requested
    if args.output:
        save_results(results, args.output)

    # Setup plotter if needed
    args.prjName = grfMD['short_name']
    plot = Plotter(args)

    if 'a' in args.showPlots:
        plot.plot_cycles(grfD, grfMD, figId=1)
    
    if 'b' in args.showPlots:
        plot.plot_voids_2d(grfD, grfMD, figId=2)
    
    if 'c' in args.showPlots:
        plot.plot_voids_3d(grfD, grfMD, figId=3)
    
    plot.display_all()
  

if __name__ == "__main__":
    main()
