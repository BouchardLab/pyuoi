
import numpy as np
import gudhi
import networkx as nx
from typing import Dict, Tuple, List

class BettiComputeres:
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


def print_betti_summary(grfD: dict, grfMD: dict, verbose: bool = True):
    """Print formatted summary of results"""
    
    print("\n" + "="*60)
    print("GRAPH BETTI NUMBER COMPUTATION SUMMARY")
    print("="*60)
    
    # Extract data from the structured dictionaries
    graph_conf = grfMD['graph_conf']
    betti_results = grfMD['betti_results']
    
    # Graph properties
    print(f"\n📊 Graph Properties:")
    print(f"   Nodes: {graph_conf['n_nodes']}")
    
    # Handle different graph types - random vs matrix-derived
    if 'edge_probability' in graph_conf:
        # Random graph case
        print(f"   Edge probability: {graph_conf['edge_probability']:.1%}")
        print(f"   Actual edges: {graph_conf['n_edges']:,}")
        print(f"   Expected edges: {int(graph_conf['n_nodes']*(graph_conf['n_nodes']-1)/2 * graph_conf['edge_probability']):,}")
    else:
        # Matrix-derived graph case
        print(f"   Actual edges: {graph_conf['n_edges']:,}")
        max_possible_edges = graph_conf['n_nodes']*(graph_conf['n_nodes']-1)//2
        print(f"   Max possible edges: {max_possible_edges:,}")
        edge_density = graph_conf['n_edges'] / max_possible_edges
        print(f"   Edge density: {edge_density:.1%}")
        
        # Show threshold if available
        if 'edge_ampl_thres' in graph_conf:
            print(f"   Amplitude threshold: {graph_conf['edge_ampl_thres']}")
        if 'data_type' in graph_conf:
            print(f"   Data type: {graph_conf['data_type']}")
    
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
    
