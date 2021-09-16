from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist

import numpy as np

import networkx as nx
import sys


def adjacentRidges(a, b):
    """Find ridges adjacent between two shapes.

    Parameters
    ----------
    a       : input 1D array
    b       : input 1D array

    Output
    ------
    Output  : 1D Array of values of vertices that
    form a shared ridge between 'a' and 'b'.
    In case there is no match it returns an empty array.
    """

    # Return values repeated in both arrays (may form shared ridge)
    values, ida, idb = np.intersect1d(
        a, b,
        assume_unique = True,
        return_indices = True
        )

    # Find distance from one another in the same array (a ridge must have a distance of just one position)
    dist_a = cdist(ida[::-1,np.newaxis], ida[::-1,np.newaxis], metric = lambda a,b: b - a)

    # Return numeric values of indices 1 distance away from each other
    return np.array(a[
                ida[
                    np.array(
                        np.isin(np.triu(dist_a, k = 1), [1,-1,1-len(a),len(a)-1]).nonzero()
                        ).T
                ]
            ])


def search_sequence_numpy(arr,seq, onlyFirst = False):
    """ Find sequence in an array using NumPy only.

    Parameters
    ----------
    arr    : input 1D array
    seq    : input 1D array

    Output
    ------
    Output : 1D Array of indices in the input array that satisfy the
    matching of input sequence in the input array.
    In case of no match, an empty list is returned.
    """

    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    ind = (np.arange(Na)[:,None] + r_seq) % Na
    M = (arr[ind] == seq).all(1)

    # Get the range of those indices as final output
    if M.any() >0:
        if onlyFirst:
            return ind[M][0,1]
        return ind[M]
        # return np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0)[0] % Na
    else:
        if onlyFirst:
            return None
        return np.array([])         # No match found


def findTips(comb):
    """Find position of tip in array sequence. A tip is a unique vertex where
    k-distant pairs of neighbours are the same until l-distance. Minimum
    distance to be considered tip is m=1.
    Ex: Array [3,4,1,0,1,2] has one tip in position 3 with value 0 and m=1
    because 1-distance pair of neighbours (i.e. 1 and 1) are the same vertex.

    Parameters
    ----------
    comb    : input 1D array of vertices

    Output
    ------
    tip     : 1D array of positions where tips are located
    tail    : 1D boolean array where repeated values are found
    If no tip is found, 'tip' and 'tail' are empty arrays.
    """

    # Delete sequences of points that shapes an empty polygon (tail)
    values, counts = np.unique(comb, return_counts = True)
    # Find values that are  repeated in the sequence
    tail = np.isin(comb, values[counts > 1])
    # No repeated values are found
    if not tail.any():
        return np.array([]), np.array([])
    # Find tips by using boolean sequence of repeated values at the tip
    tips = search_sequence_numpy(
            tail,
            np.array([True, False, True]))
    if tips.shape[0] <= 0:
        return np.array([]), np.array([])
    tip = tips[:,1]
    # Check tip is actually tip (by comparing values beside)
    sides = comb[np.array([tip-1, tip+1])]
    tip = tip[np.all(sides == sides[0,:], axis = 0)]
    return tip, tail


def delete_tail(comb):
    """Find and delete any tail found. A tail is the list of vertices extending
    from a tip up to m-distance neighbours where each k-distant pairs of
    neighbours are the same up to m-distance.

    Parameters
    ----------
    comb    : 1D input array of vertices

    Output
    ------
    Output  : 1D input array of vertices where there are no more tips.
    Caution. Deleting any tail may produce new tips not found before and this
    method may have to be invoked several times to erradicate all tails in
    'comb'.
    """

    # Find tips of any tail
    tips, tail = findTips(comb)

    # For each tip count m-distance
    coupled = np.array([0,]*len(tips))
    for i, t in enumerate(tips):
        ss = np.roll(tail, shift = -t)
        ll = np.roll(comb, shift = -t)
        for j, (s, l) in enumerate(zip(ss[1:], ll[1:])):
            if s and l == ll[-1-j]:
                coupled[i] += 1
            else:
                break

    # Do not modify 'comb' if no tip is found
    if coupled.size == 0 or tips.size == 0:
        tail_nodes = np.array([])
    else:
        # Translate m-distance to indices of the tail and tip
        tail_nodes = np.concatenate(np.array([np.arange(-c, c) + t for c, t in zip(coupled, tips)], dtype = np.ndarray))

    # Delete indices of tails
    comb = np.delete(comb, list(tail_nodes))
    return comb


def combine(a,b,pair):
    """Combine two adjacent polygons which they share at least one side (pair of
    vertices). The method breaks the first sequence in between pair values and
    inserts the second array inside. Array 'b' may be inverted in order to
    delete ridge from new formed polygon. Delete tails that may have been formed
    due to more than one ridge been shared.

    Parameters
    ----------
    a       : Ordered input 1D array
    b       : Ordered input 2D array
    pair    : Input 1D array of size 2

    Output
    ------
    Output: Ordered 1D array of vertices.
    """
    pair = np.array(list(pair))

    # When the pair of values is not a shared ridge, return the first shape
    if ~ np.isin(pair, a).all():
        return a
    # Find where pair of vertices is found at the first array
    inda, endIndex = search_sequence_numpy(a, pair)[0] \
        if search_sequence_numpy(a, pair).shape[0] != 0 \
        else search_sequence_numpy(a, pair[::-1])[0]

    # If pair is located between the begining and ending of the sequence, change
    # first index location new the end of sequence. This assures later that the
    # element right to the pair element is always the second vertex of the pair
    inda = endIndex if endIndex - inda > 1 else inda

    # Find pair in 'b' array. Second value of pair in 'b' may or may not be to
    # the right of the first one
    startIndex = a[inda]
    indb = np.isin(b,[startIndex]).nonzero()[0][0]

    a = np.roll(a, -inda)
    b = np.roll(b, -indb)

    # Invert b sequence if order of pair is the same for both arrays
    if b[1] == a[1]:
        comb = np.insert(a, 1, b[-1:1:-1])
    else:
        comb = np.insert(a, 1, b[1:-1])

    if comb.shape[0] > 3:
        # Keep deleting tails until there is no tail left
        while findTips(comb)[0].size != 0:
            comb = delete_tail(comb)

    return comb


def completePatch(S, node, flag = False):
    """Create an irregular polygon as an ordered sequence of vertices from a set
    of irregular adjacent polygons. This method is similar to the sweep line
    algorithm. From 'node' iterate all k-neighbours of node and add them to the
    final shape.

    Parameters
    ----------
    S       : NetworkX Graph object. Each node have a 'reg' parameter, a 1D
              array of vertices. Edges determine which nodes (i.e. polygons) are
              adjacent from one another (i.e. they share one ridge or a pair of
              vertices). Each edge has a 'ridge' property, a set of a pair of
              indices indicating shared  vertices beween the two nodes connected
    node    : Integer

    Output
    ------
    Output  : 1D array of ordered vertices.
    """
    # Calculate matrix distance between nodes
    paths = dict(nx.all_pairs_shortest_path_length(S))
    keys = np.array(list(paths.keys()))
    dist  = np.zeros((len(keys), len(keys)), dtype = int)
    for i,ki in enumerate(keys):
        for j, kj in enumerate(keys):
            dist[i,j] = paths[ki][kj]


    distances = np.sort(np.unique(dist.flatten()))

    # Initial shape of the S graph
    patch = S.nodes[node]['reg']
    patch_nodes = [node]

    # Sweep line algorithm
    for d in distances[1:]:
        # Search for all neighbours of d-distance from 'node'
        nextGroup = keys[
            np.array(
                (
                    np.triu(dist)[(keys == node).nonzero()[0][0],:] == d
                ).nonzero()
            ).T[:,-1]
        ]

        # Traverse for all nodes at a certain distance
        for nodeNext in nextGroup:
            # Get all neighbours from each node that have been added to the
            # patch already
            neigh = np.array(list(S.neighbors(nodeNext)))
            neigh_patch = neigh[np.isin(neigh, patch_nodes)]

            # Add node to the list of nodes that shape patch
            patch_nodes.append(nodeNext)

            # Take any neighbour node that is in shape to insert geometry into patch
            neighNode = neigh_patch[0]
            patch = combine(patch,
                S.nodes[nodeNext]['reg'],
                S.edges[nodeNext,neighNode]['ridge']
                )

    return patch


def gmap(dots, spam_points, labels):
    """Get irregular shapes of labeled data points using Voronoi diagrams for
    each point and put them together.

    Parameters
    ----------
    dots        : A 2D array of positions (nData, nDim) for each data point
    spam_points : Outliers to draw finite Voronoi shapes for data points
    labels      : A 1D array of int labels. Caution label '-1' is reserved for
                  outliers.

    Output
    ------
    patches_x   : List of 1D arrays of x coordinates for the vertices of each region
    patches_y   : List of 1D arrays of y coordinates for the vertices of each region
    lbl_regions : List of labels for each region corresponding to labels of the
                  grouped nodes.
    """

    # Stack data points and outliers in the same array
    points = np.vstack((dots, spam_points))
    labels = np.hstack((labels, [-1]*len(spam_points)))
    # Apply Voronoi diagrams
    vor = Voronoi(points = points)

    # Find label for each region according to labels of the group
    lbl_regions = np.array([0]*len(vor.regions), dtype = int)
    for i in np.sort(range(points.shape[0])):
        lbl_regions[vor.point_region[i]] = labels[i]
    regions = np.array(vor.regions, dtype = np.ndarray)[np.sort(vor.point_region)]
    lbl_regions = lbl_regions[np.sort(vor.point_region)]

    # Search diagrams that are infinite, i.e. its ridges are infinite or semi-infinite
    noDraw = \
        (lbl_regions == -1) + \
        (np.array([(region == []) or (-1 in region) for region in regions], dtype = bool))

    # Exclude infinite regions
    regions = regions[~noDraw]
    lbl_regions = lbl_regions[~noDraw]


    # Define regions as nodes with label and sequence of vertices
    G = nx.Graph()
    G.add_nodes_from(
    [(node, {'lbl' : lbl, 'reg' : np.array(region)}) for node, (lbl, region) in enumerate(zip(lbl_regions, regions))]
    )


    # Define adjency of polygons as edges
    for node in G.nodes:
        patch = G.nodes[node]['reg']
        # Check only non-checked pairs
        for otherNode in set(G.nodes) - set(list(G.nodes)[:node+1]):
            if G.nodes[node]['lbl'] != G.nodes[otherNode]['lbl']:
                continue
            # Add edge where pair of nodes share a ridge
            patchOther = G.nodes[otherNode]['reg']
            ridges = adjacentRidges(patch, patchOther)
            if ridges.shape[0] != 0:
                G.add_edge(node, otherNode, ridge = frozenset(ridges[0]))

    # Find patches as subgraphs of connected nodes
    patches = []
    lbl_patches = []
    for S in [G.subgraph(c).copy() for c in nx.connected_components(G)]:
        # Add label of patch from label of first found node of subgraph
        lbl_patches.append(S.nodes[list(S.nodes)[0]]['lbl'])
        patch = completePatch(S, list(S.nodes)[0])
        patches.append(patch)

    # Translate vertex indeces to vertex 2D positions
    patches = [vor.vertices[patch] for patch in patches]
    patches_x  = [patch[:,0] for patch in patches]
    patches_y  = [patch[:,1] for patch in patches]

    return patches_x, patches_y, lbl_patches, vor.vertices


if __name__ == '__main__':
    # Import libraries
    from bokeh.plotting import figure, show
    from bokeh.models import ColumnDataSource
    from bokeh.palettes import Spectral8

    from sklearn.cluster import KMeans

    # Example of random position data points clustered using k-means in N
    # clusters and apply method to define region borders and plot them.
    np.random.seed(42)

    # Initial parameters
    size = 100
    N = 5
    D = 2

    # Data points
    dots = np.random.rand(size,D)
    spam_points = np.empty((0,2), dtype = int)

    # Create outliers from which to define an exterior ridge
    std = cdist(dots, dots, 'euclidean').std()
    maxx = cdist(dots, dots, 'euclidean').max() *0.5
    outliers = np.random.rand(5000,2)
    outliers = maxx * (outliers[:,0]*1.0+0.4)[:, np.newaxis] * np.array([np.cos(outliers[:,1]*2*np.pi), np.sin(outliers[:,1]*2*np.pi)]).T

    # In case any ridge goes beyond outlier layer, this will prevent it from going to infinity
    outliersAway = np.random.rand(1000,2)
    outliersAway = maxx * (outliersAway[:,0]*0.1+5)[:, np.newaxis] * np.array([np.cos(outliersAway[:,1]*2*np.pi), np.sin(outliersAway[:,1]*2*np.pi)]).T

    # Center outliers and combine them all
    outliers += dots.mean(0)
    outliersAway += dots.mean(0)
    spam_points = np.concatenate([spam_points, outliers, outliersAway])
    spam_points = spam_points[(cdist(spam_points, dots, 'euclidean') > std*0.1).all(1),:]

    # Cluster data points according to location in 2d space
    kms = KMeans(n_clusters = N,
        n_init = 10,
        random_state = 42,
        verbose = 0,
        )
    kms.fit(dots)
    labels = kms.labels_

    # Get regions of space and polygon shapes
    patches_x, patches_y, lbl_regions, vertices_pos = gmap(dots, spam_points, np.array(labels))


    ##############################
    #   Graphic representation   #
    ##############################

    # Add parameters to show with hover tool
    TOOLTIPS = [
        ("id", "@id"),
        ("lbl", "@lbl"),
    ]

    # Create figure
    p = figure(tools="pan,lasso_select,box_select,wheel_zoom,hover", tooltips = TOOLTIPS,
        x_range = (0, 1),
        y_range = (0, 1),
        )


    # Object to save patch display information
    patches = ColumnDataSource(dict(
        xs = patches_x,
        ys = patches_y,
        color = [Spectral8[i] for i in lbl_regions],
        lbl = lbl_regions,
        id = list(range(len(lbl_regions)))
    ))

    # Object to display data point information
    dd = ColumnDataSource(dict(
        x = dots[:,0],
        y = dots[:,1],
        lbl = labels,
        id = list(range(len(labels)))
    ))

    # (Opcionally) display vertices which form patches
    vertices = ColumnDataSource(dict(
        x = vertices_pos[:,0],
        y = vertices_pos[:,1],
        id = np.arange(vertices_pos.shape[0]),
    ))

    # add a circle renderer with a size, color, and alpha for each data
    # (Un)comment as you desire to display
    p.circle('x', 'y', radius=0.005, color="navy", alpha=0.5, source = dd)
    # p.circle('x', 'y', radius=0.003, color="navy", alpha=0.5, source = vertices)
    # p.circle(spam_points[:,0], spam_points[:,1], radius=0.008, color="grey", alpha=0.4)
    p.patches('xs', 'ys', alpha=0.5, color = 'color', source = patches)

    p.sizing_mode = 'stretch_both'

    # Display Bokeh figure
    show(p)
