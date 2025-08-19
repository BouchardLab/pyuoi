#!/usr/bin/env python3

from __future__ import print_function,division
from toolbox.PlotterBackbone import PlotterBackbone
import numpy as np
import networkx as nx
import matplotlib.patches as patches

#...!...!..................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)         

#...!...!..................
    def plot_cycles(self, grfD, grfMD, maxItems=None, figId=1):
        """Plot graph with cycles highlighted in different colors"""
        
        figId=self.smart_append(figId)        
        fig=self.plt.figure(figId,facecolor='white', figsize=(12,8))
        ax = self.plt.gca()
        
        # Extract data from structures
        edges = grfD['edges']
        cycle_edges = grfD.get('cycle_edges', [])
        
        # Apply random sampling if maxItems is specified
        if maxItems is not None and maxItems > 0 and len(cycle_edges) > maxItems:
            import random
            cycle_edges = random.sample(cycle_edges, maxItems)
            print(f"Randomly selected {maxItems} cycles out of {len(grfD.get('cycle_edges', []))} for display")
        
        graph_conf = grfMD['graph_conf']
        betti_results = grfMD['betti_results']
        n_nodes = graph_conf['n_nodes']
        
        # Create NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        G.add_edges_from(edges)
        
        # Generate layout
        pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
        
        # Draw all nodes (smaller, no labels)
        nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=150, alpha=0.7)
        
        # Draw all edges in light gray
        nx.draw_networkx_edges(G, pos, edge_color='lightgray', alpha=0.3, width=1)
        
        # Color palette for cycles
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        # Highlight cycle edges and add cycle labels
        if cycle_edges and len(cycle_edges) > 0:
            for i, cycle in enumerate(cycle_edges):
                if i >= len(colors):
                    break
                color = colors[i % len(colors)]
                
                # Draw cycle edges
                cycle_edge_list = [(u, v) for u, v in cycle if G.has_edge(u, v)]
                if cycle_edge_list:
                    nx.draw_networkx_edges(G, pos, edgelist=cycle_edge_list, edge_color=color, width=3, alpha=0.8, style='dashed')
                    
                    # Highlight nodes in this cycle
                    cycle_nodes = set()
                    for u, v in cycle_edge_list:
                        cycle_nodes.add(u)
                        cycle_nodes.add(v)
                    nx.draw_networkx_nodes(G, pos, nodelist=list(cycle_nodes), node_color=color, node_size=200, alpha=0.8)
                    
                    # Add cycle number label to the first edge of the cycle
                    if cycle_edge_list:
                        u, v = cycle_edge_list[0]
                        edge_x = (pos[u][0] + pos[v][0]) / 2
                        edge_y = (pos[u][1] + pos[v][1]) / 2
                        self.plt.text(edge_x, edge_y, f'{i+1}', fontsize=12, fontweight='bold', 
                                ha='center', va='center', 
                                bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', edgecolor=color, linewidth=1))
        
        # Add title and info
        data_name = grfMD.get('short_name', grfMD.get('dataName', 'Unknown'))
        self.plt.title(f'Graph Cycles Visualization - {data_name}\n'
                  f'Nodes: {graph_conf["n_nodes"]}, Edges: {graph_conf["n_edges"]}, '
                  f'β₁ = {betti_results["betti_numbers"].get(1, 0)} cycles', 
                  fontsize=14, fontweight='bold')
        
        # Add legend if there are cycles
        if cycle_edges and len(cycle_edges) > 0:
            legend_elements = []
            for i in range(min(len(cycle_edges), len(colors))):
                color = colors[i % len(colors)]
                legend_elements.append(self.plt.Line2D([0], [0], color=color, lw=1, label=f'Cycle {i+1}'))
            self.plt.legend(handles=legend_elements, loc='upper right')
        
        self.plt.axis('off')
        self.plt.tight_layout()

#...!...!..................
    def plot_voids_3d(self, grfD, grfMD, maxItems=None, figId=3):
        """Plot 3D voids as tetrahedral projections"""
        
        figId=self.smart_append(figId)        
        fig=self.plt.figure(figId,facecolor='white', figsize=(12,8))
        ax = self.plt.gca()
        
        # Extract data from structures
        edges = grfD['edges']
        void_tetrahedra = grfD.get('void_tetrahedra', [])
        
        # Apply random sampling if maxItems is specified
        if maxItems is not None and maxItems > 0 and len(void_tetrahedra) > maxItems:
            import random
            void_tetrahedra = random.sample(void_tetrahedra, maxItems)
            print(f"Randomly selected {maxItems} 3D voids out of {len(grfD.get('void_tetrahedra', []))} for display")
        
        graph_conf = grfMD['graph_conf']
        betti_results = grfMD['betti_results']
        n_nodes = graph_conf['n_nodes']
        
        # Create NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        G.add_edges_from(edges)
        
        # Generate layout
        pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
        
        # Draw all nodes (smaller, no labels)
        nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=150, alpha=0.7)
        
        # Draw all edges in light gray
        nx.draw_networkx_edges(G, pos, edge_color='lightgray', alpha=0.3, width=1)
        
        # Color palette for 3D voids
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        # Highlight 3D voids
        if void_tetrahedra and len(void_tetrahedra) > 0:
            for i, tetrahedron in enumerate(void_tetrahedra):
                if i >= len(colors):
                    break
                color = colors[i % len(colors)]
                
                # Extract first valid tetrahedron
                if len(tetrahedron) > 0:
                    first_tetra = tetrahedron[0]
                    if len(first_tetra) >= 4:
                        # Tetrahedron edges (6 edges for 4 vertices)
                        tetra_edges = []
                        for j in range(4):
                            for k in range(j+1, 4):
                                tetra_edges.append((first_tetra[j], first_tetra[k]))
                        
                        # Draw tetrahedron edges with dash-dot pattern
                        for edge in tetra_edges:
                            if G.has_edge(edge[0], edge[1]):
                                nx.draw_networkx_edges(G, pos, edgelist=[edge], 
                                                     edge_color=color, width=4, alpha=0.9, 
                                                     style=(0, (3, 1, 1, 1)))  # dash-dot pattern
                        
                        # Highlight tetrahedron vertices
                        tetra_nodes = list(first_tetra)
                        nx.draw_networkx_nodes(G, pos, nodelist=tetra_nodes, node_color=color, node_size=300, alpha=0.9)
                        
                        # Add void number label to centroid
                        if all(node in pos for node in tetra_nodes):
                            centroid_x = sum(pos[node][0] for node in tetra_nodes) / 4
                            centroid_y = sum(pos[node][1] for node in tetra_nodes) / 4
                            self.plt.text(centroid_x, centroid_y, f'V3-{i+1}', fontsize=16, fontweight='bold', 
                                    ha='center', va='center', 
                                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=color, linewidth=1))
        
        # Add title and info
        data_name = grfMD.get('short_name', grfMD.get('dataName', 'Unknown'))
        self.plt.title(f'Graph 3D Voids Visualization - {data_name}\n'
                  f'Nodes: {graph_conf["n_nodes"]}, Edges: {graph_conf["n_edges"]}, '
                  f'β₃ = {betti_results["betti_numbers"].get(3, 0)} voids', 
                  fontsize=14, fontweight='bold')
        
        # Add legend if there are voids
        if void_tetrahedra and len(void_tetrahedra) > 0:
            legend_elements = []
            for i in range(min(len(void_tetrahedra), len(colors))):
                color = colors[i % len(colors)]
                legend_elements.append(self.plt.Line2D([0], [0], color=color, lw=2, 
                                                     linestyle=(0, (3, 1, 1, 1)), label=f'3D Void {i+1}'))
            self.plt.legend(handles=legend_elements, loc='upper right')
        
        self.plt.axis('off')
        self.plt.tight_layout()

#...!...!..................
    def plot_voids_2d(self, grfD, grfMD, maxItems=None, figId=2):
        """Plot 2D voids (cavities) as triangular faces"""
        
        figId=self.smart_append(figId)        
        fig=self.plt.figure(figId,facecolor='white', figsize=(12,8))
        ax = self.plt.gca()
        
        # Extract data from structures
        edges = grfD['edges']
        void_triangles = grfD.get('void_triangles', [])
        
        # Apply random sampling if maxItems is specified
        if maxItems is not None and maxItems > 0 and len(void_triangles) > maxItems:
            import random
            void_triangles = random.sample(void_triangles, maxItems)
            print(f"Randomly selected {maxItems} 2D voids out of {len(grfD.get('void_triangles', []))} for display")
        
        graph_conf = grfMD['graph_conf']
        betti_results = grfMD['betti_results']
        n_nodes = graph_conf['n_nodes']
        
        # Create NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        G.add_edges_from(edges)
        
        # Generate layout
        pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
        
        # Draw all nodes (smaller, no labels)
        nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=150, alpha=0.7)
        
        # Draw all edges in light gray
        nx.draw_networkx_edges(G, pos, edge_color='lightgray', alpha=0.3, width=1)
        
        # Color palette for voids
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        # Highlight 2D voids as filled triangles
        if void_triangles and len(void_triangles) > 0:
            for i, triangles in enumerate(void_triangles):
                if i >= len(colors):
                    break
                color = colors[i % len(colors)]
                
                # Plot each triangle in the void
                for triangle in triangles:
                    if len(triangle) >= 3:
                        # Extract triangle coordinates
                        triangle_coords = []
                        for node in triangle[:3]:
                            if node in pos:
                                triangle_coords.append(pos[node])
                        
                        if len(triangle_coords) == 3:
                            # Draw triangle as filled polygon
                            triangle_x = [p[0] for p in triangle_coords] + [triangle_coords[0][0]]
                            triangle_y = [p[1] for p in triangle_coords] + [triangle_coords[0][1]]
                            self.plt.fill(triangle_x, triangle_y, color=color, alpha=0.3)
                            
                            # Draw triangle edges
                            first_triangle = triangle[:3]
                            for j in range(3):
                                u, v = triangle[j], triangle[(j+1) % 3]
                                if G.has_edge(u, v):
                                    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], 
                                                         edge_color=color, width=3, alpha=0.8, style='dotted')
                
                # Add void number label to centroid of first triangle  
                if triangles and len(triangles[0]) >= 3:
                    first_triangle = triangles[0][:3]
                    if all(node in pos for node in first_triangle):
                        centroid_x = sum(pos[node][0] for node in first_triangle if node in pos) / 3
                        centroid_y = sum(pos[node][1] for node in first_triangle if node in pos) / 3
                        self.plt.text(centroid_x, centroid_y, f'V{i+1}', fontsize=14, fontweight='bold', 
                                ha='center', va='center', 
                                bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor=color, linewidth=1))
        
        # Add title and info
        data_name = grfMD.get('short_name', grfMD.get('dataName', 'Unknown'))
        self.plt.title(f'Graph 2D Voids Visualization - {data_name}\n'
                  f'Nodes: {graph_conf["n_nodes"]}, Edges: {graph_conf["n_edges"]}, '
                  f'β₂ = {betti_results["betti_numbers"].get(2, 0)} voids', 
                  fontsize=14, fontweight='bold')
        
        # Add legend if there are voids
        if void_triangles and len(void_triangles) > 0:
            legend_elements = []
            for i in range(min(len(void_triangles), len(colors))):
                color = colors[i % len(colors)]
                legend_elements.append(self.plt.Line2D([0], [0], color=color, lw=1, 
                                                     linestyle='dotted', label=f'2D Void {i+1}'))
            self.plt.legend(handles=legend_elements, loc='upper right')
        
        self.plt.axis('off')
        self.plt.tight_layout()