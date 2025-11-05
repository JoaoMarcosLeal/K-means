import numpy as np
from heapq import heappop, heappush


class State:
    def __init__(self, quadrados):
        self.quadrados = quadrados
        # Número e sua posição esperada na tabela
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
            for vizinho in vizinhos:
                if vizinho not in visitados:
                    pilha.append(vizinho)

        # Se chegou até aqui, não existe solução
        return -1, None

    def astr(self):
        custo_inicial = self.calcular_dist(self.quadrados)
        
        heap, visitados = [(custo_inicial, self)], set([])
        passos = 0

        while heap:
            custo, noh = heap[0]
            heappop(heap)

            if noh in visitados:
                continue

            visitados.add(noh)
            passos += 1

            if custo == 0:
                return passos, noh

            vizinhos = self.expandir()
            for vizinho in vizinhos:
                if vizinho not in visitados:
                    custo = self.calcular_dist(vizinho.quadrados)
                    heappush(heap, (custo, vizinho))

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


estado_inicial = np.array([[2, 0, 3], [1, 7, 4], [6, 8, 5]])

estado_final = {
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

estado_final_tabela = [[1, 2, 3], [8, 0, 4], [7, 6, 5]]

state_correto = State(estado_final_tabela)
state_errado = State([[2, 0, 3], [1, 7, 4], [6, 8, 5]])

print(state_errado.resolver())
