#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import sys

from toolbox.PlotterBackbone import PlotterBackbone
import numpy as np
import networkx as nx
import matplotlib.patches as patches

#............................
#............................
#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)         

#...!...!..................
    def plot_cycles(self, grfD, grfMD, figId=1):
        """Plot graph with cycles highlighted in different colors"""
        
        figId=self.smart_append(figId)        
        fig=self.plt.figure(figId,facecolor='white', figsize=(12,8))
        ax = self.plt.gca()
        
        # Extract data from structures
        edges = grfD['edges']
        cycle_edges = grfD.get('cycle_edges', [])
        void_triangles = grfD.get('void_triangles', [])
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
                color = colors[i % len(colors)]
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
        self.plt.title(f'Graph Cycles Visualization\n'
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
    def plot_voids_3d(self, grfD, grfMD, figId=3):
        """Plot 3D voids as tetrahedral projections"""
        
        figId=self.smart_append(figId)        
        fig=self.plt.figure(figId,facecolor='white', figsize=(12,8))
        ax = self.plt.gca()
        
        # Extract data from structures
        edges = grfD['edges']
        void_tetrahedra = grfD.get('void_tetrahedra', [])
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
        void3d_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'navy']
        
        # Plot 3D voids as tetrahedral wireframes (projected to 2D)
        if void_tetrahedra and len(void_tetrahedra) > 0:
            for i, tetrahedra_group in enumerate(void_tetrahedra):
                color = void3d_colors[i % len(void3d_colors)]
                
                # Plot each tetrahedron in the group
                for tetrahedron in tetrahedra_group:
                    if len(tetrahedron) == 4:
                        # Get positions of tetrahedron vertices
                        tetra_pos = [pos[node] for node in tetrahedron if node in pos]
                        if len(tetra_pos) == 4:
                            # Draw all 6 edges of the tetrahedron
                            tetra_edges = []
                            for j in range(4):
                                for k in range(j+1, 4):
                                    tetra_edges.append((tetrahedron[j], tetrahedron[k]))
                            
                            # Highlight tetrahedral edges with thick dash-dot lines
                            for edge in tetra_edges:
                                if G.has_edge(edge[0], edge[1]):
                                    nx.draw_networkx_edges(G, pos, edgelist=[edge], 
                                                         edge_color=color, width=4, alpha=0.9, 
                                                         style=(0, (3, 1, 1, 1)))  # dash-dot pattern
                            
                            # Draw tetrahedral faces as transparent polygons
                            # Each tetrahedron has 4 triangular faces
                            faces = [
                                [tetrahedron[0], tetrahedron[1], tetrahedron[2]],
                                [tetrahedron[0], tetrahedron[1], tetrahedron[3]],
                                [tetrahedron[0], tetrahedron[2], tetrahedron[3]],
                                [tetrahedron[1], tetrahedron[2], tetrahedron[3]]
                            ]
                            
                            for face in faces:
                                face_pos = [pos[node] for node in face if node in pos]
                                if len(face_pos) == 3:
                                    face_x = [p[0] for p in face_pos] + [face_pos[0][0]]
                                    face_y = [p[1] for p in face_pos] + [face_pos[0][1]]
                                    self.plt.fill(face_x, face_y, color=color, alpha=0.1, 
                                                edgecolor=color, linewidth=0.5, linestyle='--')
                
                # Add 3D void number label at centroid of first tetrahedron
                if tetrahedra_group:
                    first_tetrahedron = tetrahedra_group[0]
                    if len(first_tetrahedron) == 4:
                        centroid_x = sum(pos[node][0] for node in first_tetrahedron if node in pos) / 4
                        centroid_y = sum(pos[node][1] for node in first_tetrahedron if node in pos) / 4
                        self.plt.text(centroid_x, centroid_y, f'3D{i+1}', fontsize=16, fontweight='bold', 
                                ha='center', va='center', 
                                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=color, linewidth=1))
        
        # Add title and info
        self.plt.title(f'Graph 3D Voids Visualization\n'
                  f'Nodes: {graph_conf["n_nodes"]}, Edges: {graph_conf["n_edges"]}, '
                  f'β₃ = {betti_results["betti_numbers"].get(3, 0)} voids', 
                  fontsize=14, fontweight='bold')
        
        # Add legend if there are 3D voids
        if void_tetrahedra and len(void_tetrahedra) > 0:
            legend_elements = []
            for i in range(min(len(void_tetrahedra), len(void3d_colors))):
                color = void3d_colors[i % len(void3d_colors)]
                legend_elements.append(self.plt.Line2D([0], [0], color=color, lw=4, 
                                                     linestyle=(0, (3, 1, 1, 1)), label=f'3D Void {i+1}'))
            self.plt.legend(handles=legend_elements, loc='upper right')
        
        self.plt.axis('off')
        self.plt.tight_layout()

#...!...!..................
    def plot_voids_2d(self, grfD, grfMD, figId=2):
        """Plot 2D voids (cavities) as triangular faces"""
        
        figId=self.smart_append(figId)        
        fig=self.plt.figure(figId,facecolor='white', figsize=(12,8))
        ax = self.plt.gca()
        
        # Extract data from structures
        edges = grfD['edges']
        void_triangles = grfD.get('void_triangles', [])
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
        void_colors = ['magenta', 'cyan', 'yellow', 'lime', 'orange', 'purple', 'pink', 'brown', 'olive', 'navy']
        
        # Plot 2D voids as filled triangles
        if void_triangles and len(void_triangles) > 0:
            for i, triangle_group in enumerate(void_triangles):
                color = void_colors[i % len(void_colors)]
                
                # Plot each triangle in the group
                for triangle in triangle_group:
                    if len(triangle) == 3:
                        # Get positions of triangle vertices
                        triangle_pos = [pos[node] for node in triangle if node in pos]
                        if len(triangle_pos) == 3:
                            # Create triangle coordinates
                            triangle_x = [p[0] for p in triangle_pos] + [triangle_pos[0][0]]
                            triangle_y = [p[1] for p in triangle_pos] + [triangle_pos[0][1]]
                            
                            # Fill triangle with transparency
                            self.plt.fill(triangle_x, triangle_y, color=color, alpha=0.3, edgecolor=color, linewidth=1)
                            
                            # Highlight triangle edges
                            for j in range(3):
                                u, v = triangle[j], triangle[(j+1) % 3]
                                if G.has_edge(u, v):
                                    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], 
                                                         edge_color=color, width=3, alpha=0.8, style='dotted')
                
                # Add void number label at centroid of first triangle
                if triangle_group:
                    first_triangle = triangle_group[0]
                    if len(first_triangle) == 3:
                        centroid_x = sum(pos[node][0] for node in first_triangle if node in pos) / 3
                        centroid_y = sum(pos[node][1] for node in first_triangle if node in pos) / 3
                        self.plt.text(centroid_x, centroid_y, f'V{i+1}', fontsize=14, fontweight='bold', 
                                ha='center', va='center', 
                                bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor=color, linewidth=1))
        
        # Add title and info
        self.plt.title(f'Graph 2D Voids Visualization\n'
                  f'Nodes: {graph_conf["n_nodes"]}, Edges: {graph_conf["n_edges"]}, '
                  f'β₂ = {betti_results["betti_numbers"].get(2, 0)} voids', 
                  fontsize=14, fontweight='bold')
        
        # Add legend if there are voids
        if void_triangles and len(void_triangles) > 0:
            legend_elements = []
            for i in range(min(len(void_triangles), len(void_colors))):
                color = void_colors[i % len(void_colors)]
                legend_elements.append(patches.Patch(color=color, alpha=0.3, label=f'Void {i+1}'))
            self.plt.legend(handles=legend_elements, loc='upper right')
        
        self.plt.axis('off')
        self.plt.tight_layout()

#...!...!..................
    def plot_voids_3d(self, grfD, grfMD, figId=3):
        """Plot 3D voids as tetrahedral projections"""
        
        figId=self.smart_append(figId)        
        fig=self.plt.figure(figId,facecolor='white', figsize=(12,8))
        ax = self.plt.gca()
        
        # Extract data from structures
        edges = grfD['edges']
        void_tetrahedra = grfD.get('void_tetrahedra', [])
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
        void3d_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'navy']
        
        # Plot 3D voids as tetrahedral wireframes (projected to 2D)
        if void_tetrahedra and len(void_tetrahedra) > 0:
            for i, tetrahedra_group in enumerate(void_tetrahedra):
                color = void3d_colors[i % len(void3d_colors)]
                
                # Plot each tetrahedron in the group
                for tetrahedron in tetrahedra_group:
                    if len(tetrahedron) == 4:
                        # Get positions of tetrahedron vertices
                        tetra_pos = [pos[node] for node in tetrahedron if node in pos]
                        if len(tetra_pos) == 4:
                            # Draw all 6 edges of the tetrahedron
                            tetra_edges = []
                            for j in range(4):
                                for k in range(j+1, 4):
                                    tetra_edges.append((tetrahedron[j], tetrahedron[k]))
                            
                            # Highlight tetrahedral edges with thick dash-dot lines
                            for edge in tetra_edges:
                                if G.has_edge(edge[0], edge[1]):
                                    nx.draw_networkx_edges(G, pos, edgelist=[edge], 
                                                         edge_color=color, width=4, alpha=0.9, 
                                                         style=(0, (3, 1, 1, 1)))  # dash-dot pattern
                            
                            # Draw tetrahedral faces as transparent polygons
                            # Each tetrahedron has 4 triangular faces
                            faces = [
                                [tetrahedron[0], tetrahedron[1], tetrahedron[2]],
                                [tetrahedron[0], tetrahedron[1], tetrahedron[3]],
                                [tetrahedron[0], tetrahedron[2], tetrahedron[3]],
                                [tetrahedron[1], tetrahedron[2], tetrahedron[3]]
                            ]
                            
                            for face in faces:
                                face_pos = [pos[node] for node in face if node in pos]
                                if len(face_pos) == 3:
                                    face_x = [p[0] for p in face_pos] + [face_pos[0][0]]
                                    face_y = [p[1] for p in face_pos] + [face_pos[0][1]]
                                    self.plt.fill(face_x, face_y, color=color, alpha=0.1, 
                                                edgecolor=color, linewidth=0.5, linestyle='--')
                
                # Add 3D void number label at centroid of first tetrahedron
                if tetrahedra_group:
                    first_tetrahedron = tetrahedra_group[0]
                    if len(first_tetrahedron) == 4:
                        centroid_x = sum(pos[node][0] for node in first_tetrahedron if node in pos) / 4
                        centroid_y = sum(pos[node][1] for node in first_tetrahedron if node in pos) / 4
                        self.plt.text(centroid_x, centroid_y, f'3D{i+1}', fontsize=16, fontweight='bold', 
                                ha='center', va='center', 
                                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=color, linewidth=1))
        
        # Add title and info
        self.plt.title(f'Graph 3D Voids Visualization\n'
                  f'Nodes: {graph_conf["n_nodes"]}, Edges: {graph_conf["n_edges"]}, '
                  f'β₃ = {betti_results["betti_numbers"].get(3, 0)} voids', 
                  fontsize=14, fontweight='bold')
        
        # Add legend if there are 3D voids
        if void_tetrahedra and len(void_tetrahedra) > 0:
            legend_elements = []
            for i in range(min(len(void_tetrahedra), len(void3d_colors))):
                color = void3d_colors[i % len(void3d_colors)]
                legend_elements.append(self.plt.Line2D([0], [0], color=color, lw=2, 
                                                     linestyle=(0, (3, 1, 1, 1)), label=f'3D Void {i+1}'))
            self.plt.legend(handles=legend_elements, loc='upper right')
        
        self.plt.axis('off')
        self.plt.tight_layout()
