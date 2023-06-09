# IMPLEMENTAÇÃO DE UMA LISTA DUPLAMENTE ENCADEADA # 

class No:
    
    def __init__(self, valor, anterior=None, proximo=None):
        self.valor = valor
        self.anterior = anterior
        self.proximo = proximo
        
    def __repr__(self) -> str:
        return str(self.valor) + " -> " + str(self.proximo)

class ListaDuplamenteEncadeada:
    
    def __init__(self):
        self.inicio = None
        
    def add(self, valor):
        novo = No(valor, None, None)
        if self.inicio is None:
            self.inicio = novo
            return
        atual = self.inicio
        while atual.proximo:
            atual = atual.proximo
        atual.proximo = novo
        novo.anterior = atual

    def remove(self, valor):
        atual = self.inicio
        while atual:
            if atual.valor == valor:
                if atual.anterior:
                    atual.anterior.proximo = atual.proximo
                if atual.proximo:
                    atual.proximo.anterior = atual.anterior
                if atual == self.inicio:
                    self.inicio = atual.proximo
                return
            else:
                print("Elemento removido: " + str(self.inicio.valor))
                self.inicio = self.inicio.proximo
                if self.inicio is not None:
                    self.inicio.anterior = None

    def removeraa(self):
        removed_values = []
        curr = self.fim
        for i in range(4):
            if curr is not None:
                removed_values.append(curr.valor)
                curr = curr.anterior
            else:
                break
        if curr is not None:
            curr.proximo = None
        if self.fim is not None:
            self.fim.anterior = curr
        return removed_values

    def mostrar_conteudo(self):
        print("Conteúdo da Lista Duplamente Encadeada")
        curr = self.inicio
        while curr is not None:
            print(curr.valor)
            curr = curr.proximo

    def __repr__(self) -> str:
        return "[ " + str(self.inicio) + " ]"

lde = ListaDuplamenteEncadeada()

# ADICIONANDO 10 NÚMEROS (DE 0 A 9)
lde.add(0)
lde.add(1)
lde.add(2)
lde.add(3)
lde.add(4)
lde.add(5)
lde.add(6)
lde.add(7)
lde.add(8)
lde.add(9)

lde.mostrar_conteudo()
print(lde)

removed_values = lde.removeraa()

print("Valores removidos:", removed_values)
lde.mostrar_conteudo()
print(lde)

# LISTA DUPLAMENTE ENCADEADA ORDENADA #

class No:
    
    def __init__(self, valor, anterior=None, proximo=None):
        self.valor = valor
        self.anterior = anterior
        self.proximo = proximo
        
    def __repr__(self) -> str:
        return str(self.valor) + " -> " + str(self.proximo)

class ListaDuplamenteEncadeada:
    
    def __init__(self):
        self.inicio = None
        
    def add_ordenado(self, valor):
        novo = No(valor, None, None)
        if self.inicio is None:
            self.inicio = novo
            return
        if self.inicio.valor > valor:
            novo.proximo = self.inicio
            self.inicio.anterior = novo
            self.inicio = novo
            return
        atual = self.inicio
        while atual.proximo and atual.proximo.valor < valor:
            atual = atual.proximo
        novo.anterior = atual
        novo.proximo = atual.proximo
        if atual.proximo:
            atual.proximo.anterior = novo
        atual.proximo = novo

    def remove(self, valor):
        atual = self.inicio
        while atual:
            if atual.valor == valor:
                if atual.anterior:
                    atual.anterior.proximo = atual.proximo
                else:
                    self.inicio = atual.proximo
                if atual.proximo:
                    atual.proximo.anterior = atual.anterior
                return
            atual = atual.proximo

    def mostrar_conteudo(self):
        print("Conteúdo da Lista Duplamente Encadeada")

    def __repr__(self) -> str:
        return "[ " + str(self.inicio) + " ]"

lde = ListaDuplamenteEncadeada()

# ADICIONANDO 10 NÚMEROS ALEATORIAMENTE (DE 0 A 9)
lde.add_ordenado(5)
lde.add_ordenado(7)
lde.add_ordenado(0)
lde.add_ordenado(6)
lde.add_ordenado(1)
lde.add_ordenado(9)
lde.add_ordenado(2)
lde.add_ordenado(4)
lde.add_ordenado(8)
lde.add_ordenado(3)
# FOI ORGANIZADO EM ORDEM CRESCENTE

lde.mostrar_conteudo()
print(lde)

# REMOVENDO UM ELEMENTO COM A INSERÇÃO DE UM PARÂMETRO
lde.remove(5)

lde.mostrar_conteudo()
print(lde)

# ÁRVORE BINÁRIA #

import random

class Node:
    def __init__(self, value):
        self.value = value
        self.left_child = None
        self.right_child = None

class BinaryTree:
    def __init__(self):
        self.root = None

    def insert(self, value):
        new_node = Node(value)
        if self.root is None:
            self.root = new_node
            return
        current_node = self.root
        while True:
            if value < current_node.value:
                if current_node.left_child is None:
                    current_node.left_child = new_node
                    return
                current_node = current_node.left_child
            else:
                if current_node.right_child is None:
                    current_node.right_child = new_node
                    return
                current_node = current_node.right_child

    def size(self, node):
        if node is None:
            return 0
        return 1 + self.size(node.left_child) + self.size(node.right_child)

    def insert_random(self):
        for i in range(30):
            self.insert(random.randint(1, 100))

tree = BinaryTree()

# TESTE #

#from main import BinaryTree

def test_quantidade_arvore():
    arvore = BinaryTree()
    arvore.insert_random()
    assert arvore.size(arvore.root) == 30

# CAMINHAMENTO EM ÁRVORES BINÁRIAS #

# Python program to demonstrate
# insert operation in binary search tree
 
# A utility class that represents
# an individual node in a BST
 

class No:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key
 
# A utility function to insert
# a new node with the given key
 
class BST:
    def insert(self, root, key):
        if root is None:
            return No(key)
        else:
            if root.val == key:
                return root
            elif root.val < key:
                root.right = self.insert(root.right, key)
            else:
                root.left = self.insert(root.left, key)
        return root
     
    # A utility function to do inorder tree traversal
    def inorder(self, root, result = []):
        if root:
            self.inorder(root.left, result)
            # print(root.val)
            result.append(root.val)
            self.inorder(root.right, result)

    def preorder(self, root, result = []):
        if root:
            result.append(root.val)
            self.preorder(root.left, result)
            self.preorder(root.right, result)

    def postorder(self, root, result = []):
        if root:
            self.postorder(root.left, result)
            self.postorder(root.right, result)
            result.append(root.val)

    
bst = BST()
# Driver program to test the above functions
# Let us create the following BST
#     50
#   /   \
#  30   70
# / \   / \
#20 40 60 80
 
r = No(50)
r = bst.insert(r, 30)
r = bst.insert(r, 20)
r = bst.insert(r, 40)
r = bst.insert(r, 70)
r = bst.insert(r, 60)
r = bst.insert(r, 80)

# Print inoder traversal of the BST
mylist = [] 
bst.inorder(r, mylist)
print(mylist)

# TESTE #

#from main import BST, No

def test_inorder():
    bst = BST()
    r = No(50)
    r = bst.insert(r, 30)
    r = bst.insert(r, 20)
    r = bst.insert(r, 40)
    r = bst.insert(r, 70)
    r = bst.insert(r, 60)
    r = bst.insert(r, 80)
    mylist = [] 
    bst.inorder(r, mylist)
    assert mylist == [20, 30, 40, 50, 60, 70, 80]

def test_preorder():
    bst = BST()
    r = No(50)
    r = bst.insert(r, 30)
    r = bst.insert(r, 20)
    r = bst.insert(r, 40)
    r = bst.insert(r, 70)
    r = bst.insert(r, 60)
    r = bst.insert(r, 80)
    mylist = [] 
    bst.preorder(r, mylist)
    assert mylist == [50, 30, 20, 40, 70, 60, 80]

def test_postorder():
    bst = BST()
    r = No(50)
    r = bst.insert(r, 30)
    r = bst.insert(r, 20)
    r = bst.insert(r, 40)
    r = bst.insert(r, 70)
    r = bst.insert(r, 60)
    r = bst.insert(r, 80)
    mylist = [] 
    bst.postorder(r, mylist)
    assert mylist == [20, 40, 30, 60, 80, 70, 50]

# ÁRVORE BINÁRIA AVL #

class AVLNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
        self.height = 1

class AVLTree:
    def __init__(self):
        self.root = None
    
    def insert(self, data):
        def _insert(node, data):
            if not node:
                return AVLNode(data)
            
            if data < node.data:
                node.left = _insert(node.left, data)
            else:
                node.right = _insert(node.right, data)
            
            node.height = 1 + max(self.get_height(node.left), self.get_height(node.right))
            balance = self.get_balance(node)
            
            if balance > 1 and data < node.left.data:
                return self.rotate_right(node)
            
            if balance < -1 and data > node.right.data:
                return self.rotate_left(node)
            
            if balance > 1 and data > node.left.data:
                node.left = self.rotate_left(node.left)
                return self.rotate_right(node)
            
            if balance < -1 and data < node.right.data:
                node.right = self.rotate_right(node.right)
                return self.rotate_left(node)
            
            return node
        
        self.root = _insert(self.root, data)
    
    def search(self, data):
        def _search(node, data):
            if not node:
                return None
            
            if node.data == data:
                return node
            
            if data < node.data:
                return _search(node.left, data)
            else:
                return _search(node.right, data)
        
        return _search(self.root, data)
    
    def get_height(self, node):
        if not node:
            return 0
        else:
            return node.height
    
    def get_balance(self, node):
        if not node:
            return 0
        else:
            return self.get_height(node.left) - self.get_height(node.right)
    
    def rotate_right(self, node):
        left = node.left
        left_right = left.right
        
        left.right = node
        node.left = left_right
        
        node.height = 1 + max(self.get_height(node.left), self.get_height(node.right))
        left.height = 1 + max(self.get_height(left.left), self.get_height(left.right))
        
        return left
    
    def rotate_left(self, node):
        right = node.right
        right_left = right.left
        
        right.left = node
        node.right = right_left
        
        node.height = 1 + max(self.get_height(node.left), self.get_height(node.right))
        right.height = 1 + max(self.get_height(right.left), self.get_height(right.right))
        
        return right
    
# TESTE #

#import pytest
#from main import AVLTree

#@pytest.fixture
def tree():
    return AVLTree()

def test_insert(tree):
    tree.insert(10)
    assert tree.root.data == 10
    tree.insert(20)
    assert tree.root.right.data == 20
    tree.insert(30)
    assert tree.root.right.data == 30
    tree.insert(40)
    assert tree.root.right.right.data == 40
    tree.insert(50)
    assert tree.root.right.right.data == 50
    tree.insert(25)
    assert tree.root.left.right.data == 25
    assert tree.root.data == 30
    assert tree.root.left.left.data == 10
    assert tree.root.left.data == 20
    assert tree.root.right.data == 40
    assert tree.root.right.right.data == 50

def test_search(tree):
    tree.insert(10)
    tree.insert(20)
    assert tree.search(10).data == 10
    assert tree.search(20).data == 20

# CAMINHAMENTO LARGURA #

class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
    
class BinaryTree:
    def __init__(self):
        self.root = None
    
    def insert(self, value):
        new_node = Node(value)
        if self.root is None:
            self.root = new_node
        else:
            current_node = self.root
            while True:
                if value < current_node.value:
                    if current_node.left is None:
                        current_node.left = new_node
                        break
                    else:
                        current_node = current_node.left
                else:
                    if current_node.right is None:
                        current_node.right = new_node
                        break
                    else:
                        current_node = current_node.right
    
    def bfs_traversal(self):
        if self.root is None:
            return []
        queue = [self.root]
        result = []
        while queue:
            current_node = queue.pop(0)
            result.append(current_node.value)
            if current_node.left:
                queue.append(current_node.left)
            if current_node.right:
                queue.append(current_node.right)
        return result
    
# TESTE #

#import pytest
#from main import BinaryTree

def test_bfs_traversal():
    tree = BinaryTree()
    tree.insert(50)
    tree.insert(20)
    tree.insert(80)
    tree.insert(10)
    tree.insert(30)
    tree.insert(70)
    tree.insert(90)
    tree.insert(5)
    tree.insert(15)
    tree.insert(25)
    tree.insert(35)
    tree.insert(60)
    tree.insert(75)
    tree.insert(95)
    tree.insert(2)
    tree.insert(7)
    tree.insert(12)
    tree.insert(18)
    tree.insert(22)
    tree.insert(28)
    tree.insert(33)
    tree.insert(40)
    tree.insert(55)
    tree.insert(65)
    tree.insert(72)
    tree.insert(77)
    tree.insert(85)
    tree.insert(92)
    tree.insert(98)

    expected = [50, 20, 80, 10, 30, 70, 90, 5, 15, 25, 35, 60, 75, 85, 95, 2, 7, 12, 18, 22, 28, 33, 40, 55, 65, 72, 77, 92, 98]
    assert tree.bfs_traversal() == expected

# ÁRVORES BINÁRIAS HEAP #

# Árvore Binária Heap (Maximal)

class BinaryTreeHeap:
    def __init__(self):
        self.heap = []
        self.size = 0
        self.tail = -1

    def insert(self, value):
        self.heap.append(value)
        self.size += 1
        self._heapify_up(self.size)

    def remove(self):
        if self.size == 0:
            return None
        
        root = self.heap[1]
        self.heap[1] = self.heap[self.size]
        self.size -= 1
        self.heap.pop()
        self._heapify_down(1)

        return root

    def find_parent(self, index):
        if index == 1:
            return None
        
        parent_index = [index - 1] // 2
        return self.heap[parent_index]

    def find_left_child(self, index):
        left_child_index = [2 * index] + 1

        if left_child_index > self.size:
            return None
        
        return self.heap[left_child_index]

    def find_right_child(self, index):
        right_child_index = 2 * [index + 1]

        if right_child_index > self.size:
            return None
        
        return self.heap[right_child_index]

    # Outras funções podem vir aqui.

    def isEmpty(self):
        return self.tail
    
    def _heapify_up(self, index):
        parent_index = [index - 1] // 2
        
        if parent_index <= 0:
            return
        
        if self.heap[index] < self.heap[parent_index]:
            self.heap[index], self.heap[parent_index] = self.heap[parent_index], self.heap[index]
            self._heapify_up(parent_index)
    
    def _heapify_down(self, index):
        left_child_index = [2 * index] + 1
        right_child_index = 2 * [index + 1]
        smallest_index = index
        
        if left_child_index <= self.size and self.heap[left_child_index] < self.heap[smallest_index]:
            smallest_index = left_child_index
            
        if right_child_index <= self.size and self.heap[right_child_index] < self.heap[smallest_index]:
            smallest_index = right_child_index
        
        if smallest_index != index:
            self.heap[index], self.heap[smallest_index] = self.heap[smallest_index], self.heap[index]
            self._heapify_down(smallest_index)

# ÁRVORES BINÁRIAS HEAP COM PYTEST # 

# Implementacao da BinaryHeap maximal
class BinaryHeap:
    def __init__(self):
        self.heap = []

    def insert(self, value):
        self.heap.append(value)
        self._heap_up(len(self.heap) - 1)

    def remove(self):
        if not self.heap:
            return None
        if len(self.heap) == 1:
            return self.heap.pop()
        root = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heap_down(0)
        return root
    
    def _heap_down(self, index):
        left = 2 * index + 1
        right = 2 * index + 2
        largest = index
        if left < len(self.heap) and self.heap[left] > self.heap[largest]:
            largest = left
        if right < len(self.heap) and self.heap[right] > self.heap[largest]:
            largest = right
        if largest != index:
            self.heap[index], self.heap[largest] = self.heap[largest], self.heap[index]
            self._heap_down(largest)    
    
    def _heap_up(self, index):
        parent = (index - 1) // 2
        if index <= 0:
            return
        elif self.heap[parent] < self.heap[index]:
            self.heap[parent], self.heap[index] = self.heap[index], self.heap[parent]
            self._heap_up(parent)

    def to_list(self):
        return self.heap
    
# TESTE #

#import pytest

#from main import BinaryHeap

#@pytest.fixture
def heap():
    return BinaryHeap()

def test_insert_maximal_heap(heap):
    heap.insert(88)
    assert heap.to_list() == [88]
    heap.insert(87)
    assert heap.to_list() == [88, 87]
    heap.insert(73)
    assert heap.to_list() == [88, 87, 73]
    heap.insert(47)
    assert heap.to_list() == [88, 87, 73, 47]
    heap.insert(54)
    assert heap.to_list() == [88, 87, 73, 47, 54]
    heap.insert(6)
    assert heap.to_list() == [88, 87, 73, 47, 54, 6]
    heap.insert(0)
    assert heap.to_list() == [88, 87, 73, 47, 54, 6, 0]
    heap.insert(43)
    assert heap.to_list() == [88, 87, 73, 47, 54, 6, 0, 43]
    heap.insert(100)
    assert heap.to_list() == [100, 88, 73, 87, 54, 6, 0, 43, 47]
    heap.insert(90)
    assert heap.to_list() == [100, 90, 73, 87, 88, 6, 0, 43, 47, 54]


# GRAFO DIJKSTRA 1959 MENOR CAMINHO #

G = {'a' : [('b',60),('c',54)],
     'b' : [('a',60),('d',71),('f',29)],
     'c' : [('a',54),('d',56),('e',67)],
     'd' : [('b',71),('c',56),('e',26),('g',87)],
     'e' : [('c',67),('d',26),('g',70),('i',73)],
     'f' : [('b',29),('d',52),('g',20),('h',25)],
     'g' : [('d',87),('f',20),('e',70),('h',36),('j',32),('i',59)],
     'h' : [('f',25),('g',36),('j',25)],
     'i' : [('e',73),('g',59),('j',26)],
     'j' : [('i',26),('g',32),('h',25)]
     }

import heapq

def dijkstra(G, start):

    distancias = {v: float('inf') for v in G}
    distancias[start] = 0

    heap = [(0, start)]
    while len(heap) > 0:
        distancia_atual, v_atual = heapq.heappop(heap)

        if distancia_atual > distancias[v_atual]:
            continue

        for vizinho, weight in G[v_atual]:
            distancia = distancia_atual + weight

            if distancia < distancias[vizinho]:
                distancias[vizinho] = distancia
                heapq.heappush(heap, (distancia, vizinho))

    return {v: (distancias[v], start) if distancias[v] != float('inf') else (None, None) for v in G}

D = dijkstra(G, 'a')
print(D)

# Códigos adicionais para a prova #


class NodoArvore: # Representação de Árvores Binárias #
    def __init__(self, chave=None, esquerda=None, direita=None):
        self.chave = chave
        self.esquerda = esquerda
        self.direita = direita
        
    def __repr__(self):
        return '%s <- %s -> %s' % (self.esquerda and self.esquerda.chave, 
                                   self.chave,
                                   self.direita and self.direita.chave)
    
raiz = NodoArvore(3) # Criando nodos de uma árvore utilizando o código acima #
raiz.esquerda = NodoArvore(5)
raiz.direita = NodoArvore(1)
print("Árvore: ", raiz)
    
# Caminhamento em ordem, primeiro construiremos uma árvore binária de pesquisa #

raiz = NodoArvore(40)

raiz.esquerda = NodoArvore(20)
raiz.direita = NodoArvore(60)

raiz.direita.esquerda = NodoArvore(50)
raiz.direita.direita = NodoArvore(70)
raiz.esquerda.esquerda = NodoArvore(10)
raiz.esquerda.direita = NodoArvore(30)

# Como implementar o caminhamento em ordem #

def em_ordem(raiz):
    if not raiz:
        return
    
    # Visita filho na esquerda.
    em_ordem(raiz.esquerda)
    
    # Visita nodo corrente.
    print(raiz.chave),
    
    # Visita filho da direita.
    em_ordem(raiz.direita)
    
# Executando o código acima na árvore criada anteriormente #

em_ordem(raiz)

10
20
30
40
50
60
70

# Inserção em Árvores Binárias de Pesquisa #

def insere(raiz, nodo):
    """Insere um nodo em uma árvore binária de pesquisa."""
    # Nodo deve ser inserido na raiz.
    if raiz is None:
        raiz = nodo
        
    # Nodo deve ser inserido na subárvore direita.
    elif raiz.chave < nodo.chave:
        if raiz.direita is None:
            raiz.direita = nodo
        else:
            insere(raiz.direita, nodo)
            
    # Nodo deve ser inserido na subárvore esquerda.
    else:
        if raiz.esquerda is None:
            raiz.esquerda = nodo
        else: 
            insere(raiz.esquerda, nodo)
            
# Função de inserção que acabamos de desenvolver #

# Cria uma árvore binária de pesquisa.
raiz = NodoArvore(40)
for chave in [20, 60, 50, 70, 10, 30]:
    nodo = NodoArvore(chave)
    insere(raiz, nodo)
# Imprime o caminhamento em ordem da árvore.
em_ordem(raiz)

10
20
30
40
50
60
70

# Busca de Árvores Binárias em Pesquisa #
    
def busca(raiz, chave):
    """Procura por uma chave em uma árvore binária de pesquisa."""
    #Trata o caso em que a chave procurada não está presente.
    if raiz is None:
        return None
    
    # A chave procurada está na raiz da árvore.
    if raiz.chave == chave:
        return raiz
    
    # A chave procurada é maior que a da raiz.
    if raiz.chave < chave:
        return busca(raiz.direita, chave)
    
    # A chave procurada é menor que a da raiz.
    return busca(raiz.esquerda, chave)

# O código abaixo testa nossa implementação de busca em árvores binárias de pesquisa #

# Cria uma árvore binária de pesquisa.
raiz = NodoArvore(40)
for chave in [-50, 10, 30, 70, 100]:
    resultado = busca(raiz, chave)
    if resultado:
        print("Busca pela chave {}: Sucesso!".format(chave))
    else:
        print("Busca pela chave {}: Falha!".format(chave))
        
Busca pela chave -50: Falha!
Busca pela chave 10: Sucesso!
Busca pela chave 30: Sucesso!
Busca pela chave 70: Sucesso!
Busca pela chave 100: Falha!

# Mais códigos #

class BSTNode(object):
    def __init__(self, key, value=None, left=None, right=None):
        self.key = key
        self.value = value
        self.left = left
        self.right = right
        
# Construção de um nó, com chave 42
root = BSTNode(42) 
# 42

# Adicionando filhos ao nó 42
root.left = BSTNode(10)
root.right = BSTNode(90)
#    42
#   /  \
# 10    90

# Adicionando um filho esquerdo ao nó de valor 10
root.left.left = BSTNode(2)
#       42
#      /  \
#    10    90
#   / 
#  2  
    
    def get(self, key):
        if key < self.key:
            return self.left.get(key) if self.left else None
        elif key > self.key:
            return self.right.get(key) if self.right else None
        else:
            return self
        
# Para realizar uma busca pela chave 4

tree = BSTNode(8)
...
found = tree.get(4)
if found:
    print(found)

# O método get apresentado acima poderia ser refatorado, evitando a double código #

def get(self, key):
    """Retorna uma referência ao nó de chave key"""
    if self.key == key:
        return self
    node = self.left if key < self.key else self.right
    if node is not None:
        return node.get(key)
    
# Método de inserção #

    def add(self, node):
        if node.value < self.value:
            if self.left is None:
                self.left = node
            else:
                self.left.add(node)
        else:
            if self.right is None:
                self.right = node
            else:
                self.right.add(node)
                
# Refatorando para diminuir #

    def add(self, key):
    """Adiciona elemento à subárvore"""
    side = 'left' if key < self.key else 'right'
    node = getattr(self, side)
    if node is None:
        setattr(self, side, BSTNode(key))
    else:
        node.add(key) 
       
tree = BSTNode(0)
for i in range(1, 10):
    tree.add(i)
    
# Remover, mínimo, remover mínimo #

    def remove(self, key):
        if key < self.key:
            self.left = self.left.remove(key)
        elif key > self.key:
            self.right = self.right.remove(key)
        else:
            # encontramos o elemento, então vamos removê-lo!
            if self.right is None:
                return self.left
            if self.left is None:
                return self.right
            #ao invés de remover o nó, copiamos os valores do nó substituto
            tmp = self.right._min()
            self.key, self.value = tmp.key, tmp.value
            self.right._remove_min()
        return self
 
    def _min(self):
        """Retorna o menor elemento da subárvore que tem self como raiz.
        """
        if self.left is None:
            return self
        else:
            return self.left._min()
 
    def _remove_min(self):
        """Remove o menor elemento da subárvore que tem self como raiz.
        """
        if self.left is None:  # encontrou o min, daí pode rearranjar
            return self.right
        self.left = self.left._removeMin()
        return self
    
# Método que implementa as 3 possíveis estratégias para visitar todos os nós da árvore #

    def traverse(self, visit, order='pre'):
    """Percorre a árvore na ordem fornecida como parâmetro (pre, pos ou in)
       visitando os nós com a função visit() recebida como parâmetro.
    """
    if order == 'pre':
        visit(self)
    if self.left is not None:
        self.left.traverse(visit, order)
    if order == 'in':
        visit(self)
    if self.right is not None:
        self.right.traverse(visit, order)
    if order == 'post':
        visit(self)

tree.traverse(print, 'in')
