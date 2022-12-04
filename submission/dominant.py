import sys, os, time
import networkx as nx
import numpy as np
from copy import deepcopy
import time

def is_dominant(graph, node_set):
    '''Détermine si l'ensemble de noeud node_set domine ou pas le graphe et les noeuds non dominés le cas échéant.
    Renvoie un tuple dont le premier élément est un booléen qui indique si node_set domine le graphe.
    Le deuxième élément renvoyé est l'ensemble des noeuds non dominés par node_set, si node_set est dominant, renvoie l'ensemble vide 

    Arguments :
    graph -- graphe
    node_set -- ensemble considéré
    '''
    # Ensemble supposé dominant
    dominating_set = set(node_set)
    # Ensemble des nœuds dominants et dominés
    dominated_nodes = dominating_set.copy()

    # Pour chaque noeuds dominants
    for node in dominating_set:
        # On ajoute à l'ensemble des noeuds dominés l'ensemble des voisins du nœud dominant
        dominated_nodes.update(graph.neighbors(node))

    # Ensemble des noeuds non encore dominés
    non_dominated_nodes = set(graph.nodes()) - dominated_nodes
    return len(non_dominated_nodes) == 0, non_dominated_nodes


def weight(graph, node_set):
    '''Renvoie la somme cumulée des poids des noeuds de l'ensemble node_set

    Arguments :
    graph -- graphe
    node_set -- ensemble considéré
    '''
    sum = 0
    for node in node_set:
        sum += graph.nodes[node]["weight"]
    return sum


def greedy(graph):
    '''Algorithme glouton qui construit et renvoie un ensemble dominant du graphe grâce à une heuristique donnée.
    L'heuristique préfère les noeuds qui de poids faible avec un grand nombre de noeuds voisins dont la somme des poids est élevée.

    Arguments :
    graph -- graphe
    '''

    # Ensemble des noeuds non dominés
    non_dominated_set = set(graph.nodes)
    # Ensemble des noeuds non dominants candidats
    candidates = {n: {n} | set(graph.neighbors(n)) for n in graph}

    # Ensemble des noeuds dominants
    dominating_set = set()
    # Ensemble des noeuds dominés
    dominated_set = set()

    # Tant que tous les noeuds du graphe ne sont pas encore dominés
    while non_dominated_set:

        best_score, next = np.inf, None

        # On parcourt tous les noeuds non dominants
        for node, neighbors in candidates.items():
            
            # Ensemble des voisins non encodre dominés
            coverage = list((neighbors - dominated_set))
            score = np.inf if len(coverage) == 0 else graph.nodes[node]["weight"] / len(coverage)

            if score < best_score:
               best_score = score  
               next = node

        dominating_set.add(next)
        
        # On ajoute le noeud dominant et ses voisins dans l'ensemble des noeuds dominés
        dominated_set = set.union(dominated_set, {next}, candidates[next])
        # On enlève el noeuds dominants et ses voisins dominés de la liste des noeuds non dominés
        non_dominated_set = non_dominated_set - {next} - candidates[next] 
        # On enlève le noeud choisi des candidats non dominants potentiels
        del candidates[next]

    return dominating_set


def greedy2(graph):
    '''Algorithme glouton qui construit et renvoie un ensemble dominant du graphe grâce à une heuristique donnée.
    L'heuristique préfère les noeuds qui de poids faible avec un grand nombre de noeuds voisins dont la somme des poids est élevée.

    Arguments :
    graph -- graphe
    '''

    # Ensemble des noeuds non dominés
    non_dominated_set = set(graph.nodes)
    # Ensemble des noeuds non dominants candidats
    candidates = {n: {n} | set(graph.neighbors(n)) for n in graph}

    # Ensemble des noeuds dominants
    dominating_set = set()
    # Ensemble des noeuds dominés
    dominated_set = set()

    # Tant que tous les noeuds du graphe ne sont pas encore dominés
    while non_dominated_set:

        best_score, next = np.inf, None

        # On parcourt tous les noeuds non dominants
        for node, neighbors in candidates.items():
            
            # Ensemble des voisins non encodre dominés
            coverage = list((neighbors - dominated_set))

            w=0
            for neigh in coverage:
                w+=graph.nodes[neigh]["weight"]

            score = np.inf if w == 0 else graph.nodes[node]["weight"] / w

            if score < best_score:
               best_score = score  
               next = node

        dominating_set.add(next)
        
        # On ajoute le noeud dominant et ses voisins dans l'ensemble des noeuds dominés
        dominated_set = set.union(dominated_set, {next}, candidates[next])
        # On enlève el noeuds dominants et ses voisins dominés de la liste des noeuds non dominés
        non_dominated_set = non_dominated_set - {next} - candidates[next] 
        # On enlève le noeud choisi des candidats non dominants potentiels
        del candidates[next]

    return dominating_set


def get_2_lvl_neighbors(graph, node):
    ''' Renvoie l'ensemble des voisins de second degré du noeud en paramètre.
    On entend par voisins de second degré l'enemble des voisins du neud ainsi que les voisins des voisins. 

    Arguments :
    graph -- graphe
    node -- noeud considéré
    '''
    # Ensemble des voisins de second degré (sans doublons)
    two_lvl_neighbors = set()

    # Parcourt les voisins du noeuds
    for one_lvl_neighbor in graph.neighbors(node):
        two_lvl_neighbors.add(one_lvl_neighbor)
        # Pour chaque voisin, parcourt les voisins du voisin
        for two_lvl_neighbor in graph.neighbors(one_lvl_neighbor):
            two_lvl_neighbors.add(two_lvl_neighbor)

    # On enlève le noeud qui ne fait n'est pas inclus dans ses voisins du second degré
    two_lvl_neighbors.remove(node)
    return two_lvl_neighbors


def compute_score(graph, node, node_set, freq):
    """ Fonction de scoring qui prend en compte la fréquence des sommets, qui peut être considérée comme une sorte de
    d'information dynamique indiquant l'efficacité cumulée de la recherche sur le sommet.
    Intuitivement, si un sommet est généralement non dominé, alors on essaye d'encourager l'algorithme à sélectionner 
    ce sommet pour le rendre dominé.
    Le score d'un sommet indique le bénéfice (positif ou négatif) produit en ajoutant (ou en supprimant)
    un sommet à la solution candidate.

    Arguments :
    graph -- graphe
    node -- noeud considéré
    node_set -- ensmeble des noeuds dominants
    freq -- tableau de fréquence des noeuds du graph
    """
    # Score initialization
    s = 0

    # Noeuds non dominés avec la solution actuelle node_set
    uncovered = is_dominant(graph, node_set)[1]

    # Si le noeud est dans la solution dominante
    if node not in node_set:
        # On calcule le score en sommant les fréquences des noeuds non dominés 
        # qui deviendrait dominés en ajoutant le noeud node à la solution node_set
        new_node_set = deepcopy(node_set)
        new_node_set.add(node)
        new_uncovered = is_dominant(graph, new_node_set)[1]

        nodes = uncovered - new_uncovered
        for n in nodes:
            s+= freq[n]

    # Si le noeud n'est pas dans la solution dominante
    else:
        # On calcule le score en sommant les fréquences des noeuds dominés 
        # qui deviendrait non dominés en enlevant le noeud node de la solution node_set
        new_node_set = deepcopy(node_set)
        new_node_set.remove(node)
        new_uncovered = is_dominant(graph, new_node_set)[1]

        nodes = new_uncovered - uncovered
        for n in nodes:
            s-= freq[n]
     
    return s/graph.nodes[node]["weight"]


def local_search(graph, cutoff=100):
    """ Algorithme de recherche locale pour le problème de l'ensemble dominant pondéré.
    Cet algorithme peut être divisé en deux parties : 
    1. Construction d'une solution à l'aide d'une heuristique gloutonne
    2. Amélioration de cette solution grâce à une recherche locale des solutions voisines.
    Après avoir construit une solution candidate initiale, l'algorithme fonctionne de manière itérative en supprimant 
    certains sommets et en y ajoutant d'autres jusqu'à ce que la limite de temps soit atteinte. 
    Enfin, la meilleure solution trouvée est renvoyée.

    Arguments :
    graph -- graphe
    node -- nombre d'itérations
    """

    ### INITIALISATION ###
    
    # Tableau pour indiquer si le 2ème niveau des voisins d'un sommet a changé (True) ou non (False)
    has_moved =  [True for _ in range(len(graph.nodes))]
    freq = [1 for _ in range(len(graph.nodes))]
    age = [0 for _ in range(len(graph.nodes))]

    ### CONSTRUCTION D'UNE SOLUTION GLOUTONNE ###
    s = greedy(graph)
    s_best = deepcopy(s)
    elapsed_time = 0

    ### RECHERCHE LOCALE ###
    while elapsed_time < cutoff:
        # Incrémentation du compteur d'itération
        elapsed_time += 1

        # Si la solution intermédiaire est dominante
        if is_dominant(graph, s)[0]:
            if weight(graph, s) < weight(graph, s_best):
                s_best = deepcopy(s) # On remplace la meilleur solution
            
            # Selection du noeud à enlever qui a le score le plus élevé
            best_score, next = -np.inf, None
            for node in s:
                # Fonction de scoring basée sur la fréquence des noeuds
                node_score = compute_score(graph, node, s, freq)
                # Selection du noeud le plus vieux parmi les noeuds aux scores les plus hauts
                if (node_score > best_score) or (node_score == best_score and next and age[node] < age[next]):
                    best_score = node_score
                    next = node

            s.remove(next) # Enlever le meilleur noeud
            # Mise à jour du tableau has_moved
            has_moved[next] = False
            for neighbor in get_2_lvl_neighbors(graph, next):
                has_moved[neighbor] = True
            # Mise à jour de l'âge du noeud
            age[next] += 1

        # La solution courante n'est plus un ensemble dominant

        # Mise à jour des scores et sélection d'un neoud de la solution à enlever
        best_score, next = -np.inf, None
        for node in s:
            node_score = compute_score(graph, node, s, freq)
            if (node_score > best_score) or (node_score == best_score and next and age[node] < age[next]):
                best_score = node_score
                next = node
        s.remove(next)
        age[next] += 1
        has_moved[next] = False
        for neighbor in get_2_lvl_neighbors(graph, next):
            has_moved[neighbor] = True

        # Initialisation de la liste tabou
        forbid_list = list()

        # Ajout de noeuds dans S tant que ce n'est pas un ensmeble dominant
        while not is_dominant(graph, s)[0]:
            # Pour un sommet, il est interdit d'être ajouté dans la solution candidate si sa configuration
            # (ensemble de ses voisins de 2nd degré) n'a pas été modifiée
            # On choisit aussi un noeud qui n'est pas dans la forbid_list : 
            # nous ne pouvons pas ajouter un nœud que nous avons déjà choisi auparavant
            allowed = [k for k in graph.nodes if k not in s and has_moved[k] and k not in forbid_list]

            best_score, next = -np.inf, None
            for node in allowed:
                node_score = compute_score(graph, node, s, freq)
                if (node_score > best_score) or (node_score == best_score and next and age[node] < age[next]):
                    best_score = node_score
                    next = node
            s.add(next) # Ajout du meilleur noeud à la solution 
            age[next] += 1
            for neighbor in get_2_lvl_neighbors(graph, next): 
                has_moved[neighbor] = True
            # Mise à jour de la liste tabou
            forbid_list.append(next)

            updated = is_dominant(graph, s)[1]
            for node in updated:
                freq[node] += 1 # Mise à jour des fréquences       

    return s_best


def dominant(graph):
    """
    A Faire:
    - Ecrire une fonction qui retourne le dominant du graphe non dirigé g passé en parametre.
    - cette fonction doit retourner la liste des noeuds d'un petit dominant de g

    :param g: le graphe est donné dans le format networkx : https://networkx.github.io/documentation/stable/reference/classes/graph.html
    """

    dominating_set = local_search(graph, cutoff=19)
    return dominating_set

#########################################
#### Ne pas modifier le code suivant ####
#########################################


def load_graph(name):
    with open(name, "r") as f:
        state = 0
        G = None
        for l in f:
            if state == 0:  # Header nb of nodes
                state = 1
            elif state == 1:  # Nb of nodes
                nodes = int(l)
                state = 2
            elif state == 2:  # Header position
                i = 0
                state = 3
            elif state == 3:  # Position
                i += 1
                if i >= nodes:
                    state = 4
            elif state == 4:  # Header node weight
                i = 0
                state = 5
                G = nx.Graph()
            elif state == 5:  # Node weight
                G.add_node(i, weight=int(l))
                i += 1
                if i >= nodes:
                    state = 6
            elif state == 6:  # Header edge
                i = 0
                state = 7
            elif state == 7:
                if i > nodes:
                    pass
                else:
                    edges = l.strip().split(" ")
                    for j, w in enumerate(edges):
                        w = int(w)
                        if w == 1 and (not i == j):
                            G.add_edge(i, j)
                    i += 1

        return G


#########################################
#### Ne pas modifier le code suivant ####
#########################################
if __name__ == "__main__":
    input_dir = os.path.abspath(sys.argv[1])
    output_dir = os.path.abspath(sys.argv[2])

    # un repertoire des graphes en entree doit être passé en parametre 1
    if not os.path.isdir(input_dir):
        print(input_dir, "doesn't exist")
        exit()

    # un repertoire pour enregistrer les dominants doit être passé en parametre 2
    if not os.path.isdir(output_dir):
        print(input_dir, "doesn't exist")
        exit()

        # fichier des reponses depose dans le output_dir et annote par date/heure
    output_filename = 'answers_{}.txt'.format(time.strftime("%d%b%Y_%H%M%S", time.localtime()))
    output_file = open(os.path.join(output_dir, output_filename), 'w')

    start = time.time()
    for graph_filename in sorted(os.listdir(input_dir)):
        # importer le graphe
        g = load_graph(os.path.join(input_dir, graph_filename))

        # calcul du dominant
        D = sorted(dominant(g), key=lambda x: int(x))

        # ajout au rapport
        output_file.write(graph_filename)
        for node in D:
            output_file.write(' {}'.format(node))
        output_file.write('\n')

    output_file.close()
    lapse = time.time() -start

    print("Total duration: ", lapse, "s")
