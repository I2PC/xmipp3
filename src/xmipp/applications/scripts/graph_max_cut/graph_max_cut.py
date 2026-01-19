#!/usr/bin/env python

# ***************************************************************************
# * Authors:     Oier Lauzirika Zarrabeitia (oierlauzi@bizkaia.eu)
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'xmipp@cnb.csic.es'
# ***************************************************************************/

from typing import Tuple, Set

import argparse
import pickle
import numpy as np
import scipy.sparse
import scipy.linalg
import cvxpy as cp
import networkx as nx

def maximum_cut_sdp(graph: scipy.sparse.csr_matrix,
                    solver: str = 'SCS' ) -> Tuple[float, Tuple[Set[int], Set[int]]]:
    n_vertices = graph.shape[0]
    
    # Remove the symmetric part of the graph so
    # that edges are only considrered once
    graph = scipy.sparse.tril(graph, format='csr')
    
    # Obtain the edges and their weights
    edges = graph.nonzero()
    weights = graph[edges].A1
    
    # Define the objective function and constraints
    P = cp.Variable(graph.shape, symmetric=True)
    constraints = [P >> 0, cp.diag(P) == 1] # SDP and diagonal of 1s
    objective_function = cp.scalar_product(weights, 1.0-P[edges])
    
    # Optimize
    problem = cp.Problem(cp.Maximize(objective_function), constraints)
    value = problem.solve(solver)
    
    # Quantize the result
    generator = np.random.default_rng()
    v = scipy.linalg.sqrtm(P.value)
    u = generator.normal(size=n_vertices)
    projections = v @ u
    
    # Generate the output
    v0 = set(np.nonzero(projections > 0)[0])
    v1 = set(range(n_vertices)) - v0 # Complement
    
    return (value, (v0, v1))

def maximum_cut_greedy(graph: scipy.sparse.csr_matrix):
    g = nx.from_scipy_sparse_array(graph)
    return nx.approximation.one_exchange(g, weight='weight')

def main(input_graph_path: str,
         output_partition_path: str,
         use_sdp: bool ):
    # Read the graph from disk
    graph = scipy.sparse.load_npz(input_graph_path)

    # Compute the cut
    if use_sdp:
        cut = maximum_cut_sdp(graph)
    else:
        cut = maximum_cut_greedy(graph)
    
    # Save the result
    with open(output_partition_path, 'wb') as f:
        pickle.dump(cut, f)
    
    
    
if __name__ == '__main__':
    # Define the input
    parser = argparse.ArgumentParser(prog='Maximum cut of a graph')
    parser.add_argument('-i', required=True)
    parser.add_argument('-o', required=True)
    parser.add_argument('--sdp', action='store_true')

    # Parse
    args = parser.parse_args()

    # Run the program
    main(
        input_graph_path=args.i,
        output_partition_path=args.o,
        use_sdp=args.sdp
    )