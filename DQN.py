import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from ambiente import AmbienteDiezMil
from utils import JUGADA_PLANTARSE, JUGADA_TIRAR, RANGOS


class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class AgenteDQN:
    """Agente que implementa el algoritmo de Deep Q-Learning."""

    def __init__(
        self,
        ambiente: AmbienteDiezMil,
        alpha: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ):
        """Definir las variables internas de un Agente que implementa el algoritmo de DQN.

        Args:
            ambiente (AmbienteDiezMil): Ambiente con el que interactuará el agente.
            alpha (float): Tasa de aprendizaje.
            gamma (float): Factor de descuento.
            epsilon (float): Probabilidad de explorar.
        """
        self.ambiente = ambiente
        self.alpha = alpha  
        self.gamma = gamma  
        self.epsilon = epsilon

        input_dim = 2  # PUNTOS TURNO + DADOS DISPONIBLES
        output_dim = 2  # Número de acciones: plantarse o tirar
        self.q_network = DQNetwork(input_dim, output_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()

        self.posibles_acciones = [JUGADA_PLANTARSE, JUGADA_TIRAR]

    def elegir_accion(self, estado):
        """Selecciona una acción de acuerdo a una política ε-greedy."""

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.posibles_acciones)
        else:
            estado_tensor = torch.tensor(estado, dtype=torch.float32)
            q_values = self.q_network(estado_tensor)
            accion_idx = torch.argmax(q_values).item()
            return self.posibles_acciones[accion_idx]

    def entrenar(self, episodios: int, verbose: bool = False) -> None:
        """Dada una cantidad de episodios, se repite el ciclo del algoritmo de DQN."""

        iterator = tqdm(range(episodios)) if verbose else range(episodios)
        for _ in iterator:
            self.ambiente.reset()
            juego_finalizado = False

            while not juego_finalizado:
                estado = self.ambiente.estado()
                estado_tensor = torch.tensor(estado, dtype=torch.float32)

                accion = self.elegir_accion(estado)
                recompensa, turno_finalizado, juego_finalizado = self.ambiente.step(accion)

                estado_siguiente = self.ambiente.estado()
                estado_siguiente_tensor = torch.tensor(estado_siguiente, dtype=torch.float32)

                # Predicciones Q para el estado actual y la acción tomada
                q_actual = self.q_network(estado_tensor)[self.posibles_acciones.index(accion)]

                # Calcular el valor Q target
                with torch.no_grad():
                    q_siguiente = torch.max(self.q_network(estado_siguiente_tensor))
                q_target = torch.tensor(recompensa, dtype=torch.float32) + self.gamma * q_siguiente

                # Calcular el error y actualizar la red
                loss = self.loss_fn(q_actual, q_target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def guardar_politica(self, filename: str):
        """Almacena los pesos de la red neuronal entrenada."""
        torch.save(self.q_network.state_dict(), "politicas/" + filename)
