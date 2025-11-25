import numpy as np
from heapq import heappop, heappush


class State:
    def __init__(self, quadrados):
        self.quadrados = quadrados
        # Número e sua posição esperada na tabela
        self.h = 0
        self.g = 0
        self.f = 0
        self.estado_final = {
            0: (1, 1),
            1: (0, 0),
            2: (0, 1),
            3: (0, 2),
            4: (1, 2),
            5: (2, 2),
            6: (2, 1),
            7: (2, 0),
            8: (1, 0),
        }

    def __repr__(self):
        s = ""
        for i in range(len(self.quadrados)):
            for j in range(len(self.quadrados[i])):
                s += "{} ".format(self.quadrados[i][j])
            s += "\n"
        return s

    def __eq__(self, outro):
        if isinstance(outro, State):
            # Checa se todas as posição são equivalentes
            return all(
                self.quadrados[i] == outro.quadrados[i]
                for i in range(len(self.quadrados))
            )
        return False

    def __hash__(self):
        placa_flat = tuple(item for sublist in self.quadrados for item in sublist)
        return hash(placa_flat)

    def __lt__ (self, outro):
        if isinstance(outro, State):
            return outro.f > self.f   
        return False
    
    def copia(self):
        nova_tabela = []

        for i in range(len(self.quadrados)):
            nova_tabela.append([])
            for j in range(len(self.quadrados)):
                nova_tabela[i].append(self.quadrados[i][j])

        return State(nova_tabela)

    def expandir(self):
        N = len(self.quadrados)
        vizinhos = []
        col_vazio = 0
        lin_vazio = 0

        for i in range(N):
            for j in range(N):
                if self.quadrados[i][j] == 0:
                    col_vazio = j
                    lin_vazio = i

        for [i, j] in [
            [lin_vazio + 1, col_vazio],
            [lin_vazio - 1, col_vazio],
            [lin_vazio, col_vazio + 1],
            [lin_vazio, col_vazio - 1],
        ]:
            if i >= 0 and i < N and j < N and j >= 0:
                vizinho = self.copia()
                vizinho.pai = self
                vizinho.g += self.g + 1
                vizinho.quadrados[lin_vazio][col_vazio] = self.quadrados[i][j]
                vizinho.quadrados[i][j] = self.quadrados[lin_vazio][col_vazio]
                vizinhos.append(vizinho)

        return vizinhos

    def acabou(self):
        """Checa se a tabela chegou ao seu estado final e o puzzle foi resolvido"""
        for i in range(len(self.quadrados)):
            for j in range(len(self.quadrados[i])):
                num = self.quadrados[i][j]
                lin, col = self.estado_final[num]

                if col != j or i != lin:
                    return False

        return True

    def dfs(self):
        pilha, visitados = [self], set([])
        passos = 0

        while pilha:
            noh = pilha.pop()

            if noh in visitados:
                continue

            visitados.add(noh)
            passos += 1

            if noh.acabou():
                return passos, noh

            vizinhos = noh.expandir()
            for vizinho in reversed(vizinhos): 
                if vizinho not in visitados:
                    pilha.append(vizinho)

        # Se chegou até aqui, não existe solução
        return -1, None

    def astr(self):
        self.g = 0
        self.h = self.calcular_dist(self.quadrados)
        self.f = self.h + self.g 

        g_custo = {self: self.g}
        heap = [(self.f, self)]
        passos = 0

        while heap:
            custo, noh = heap[0]
            heappop(heap)
            passos += 1

            if noh.acabou():
                return passos, noh

            if custo > g_custo.get(noh, float('inf')) + noh.h:
                    continue

            vizinhos = noh.expandir()
            for vizinho in vizinhos:
                novo_g = noh.g + 1
                
                if novo_g < g_custo.get(vizinho, float('inf')):
                    
                    vizinho.g = novo_g
                    vizinho.h = self.calcular_dist(vizinho.quadrados) 
                    vizinho.f = vizinho.g + vizinho.h                   
                    
                    g_custo[vizinho] = novo_g
                    
                    heappush(heap, (vizinho.f, vizinho))
        return -1, None
            

    def calcular_dist(self, estado):
        custo_total = 0
        for i in range(len(estado)):
            for j in range(len(estado)):
                num = estado[i][j]
                col, lin = self.estado_final[num]
                custo_total += self.get_manhattam([i, j], [col, lin])

        return custo_total

    def get_manhattam(self, point_a, point_b):
        point_a = np.array(point_a)
        point_b = np.array(point_b)
        return np.sum(np.abs(point_a - point_b))

def menu_op():
    menu = "1 - Resolver o jogo com algoritmo cego"
    menu += "\n2 - Resolver o jogo com algoritmo informado" 
    menu += "\n0 - sair\n" 
    return menu

op = int(input(menu_op())) 
while op != 0: 
    estado_inicial = []
   
    for i in range(3): 
        nums = input(f"Digite os números da linha {i + 1}: ")
        linha = list(map(int, nums.split())) 
        estado_inicial.append(linha)
    
    estado = State(estado_inicial)

    if op == 1:
        print(estado.dfs())
    if op == 2: 
        print(estado.astr())

    op = int(input(menu_op()))
