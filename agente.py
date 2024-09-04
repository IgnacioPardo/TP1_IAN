import numpy as np
from tqdm import tqdm
from ambiente import AmbienteDiezMil
from utils import JUGADA_PLANTARSE, JUGADA_TIRAR


class AgenteQLearning:
    """Agente que implementa el algoritmo de Q-Learning."""

    def __init__(
        self,
        ambiente: AmbienteDiezMil,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        # *args,
        # **kwargs,
    ):
        """Definir las variables internas de un Agente que implementa el algoritmo de Q-Learning.

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

        forma = (
            len(self.ambiente.range_snaper.rangos),  # PUNTOS TURNO
            7,  # DADOS DISPONIBLES
            2,  # ACCIONES
        )
        q_init_scale = 1
        self.init_q_table(forma, q_init_scale)
        self.posibles_acciones = [JUGADA_PLANTARSE, JUGADA_TIRAR]

    def init_q_table(self, shape, q_init_scale: float = 1):
        """
        Inicializa la tabla Q con valores aleatorios.

        Args:
            shape (tuple): Dimensiones de la tabla Q.
            q_init_scale (float): Escala para los valores iniciales de la tabla
        """
        self.q_table = np.random.uniform(0, 1, shape) * q_init_scale
        # self.q_table = np.zeros(shape)

    def elegir_accion(self):
        """
        Selecciona una acción de acuerdo a una política ε-greedy.
        """

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.posibles_acciones)
        else:
            # Select either self.posibles_acciones[0] or self.posibles_acciones[1] based on the Q table
            accion_1 = (self.ambiente.estado()) + (self.posibles_acciones[0],)
            accion_2 = (self.ambiente.estado()) + (self.posibles_acciones[1],)
            if self.q_table[accion_1] >= self.q_table[accion_2]:
                return self.posibles_acciones[0]
            else:
                return self.posibles_acciones[1]

    def entrenar(self, episodios: int, verbose: bool = False) -> None:
        """Dada una cantidad de episodios, se repite el ciclo del algoritmo de Q-learning.
        Recomendación: usar tqdm para observar el progreso en los episodios.

        Args:
            episodios (int): Cantidad de episodios a iterar.
            verbose (bool, optional): Flag para hacer visible qué ocurre en cada paso. Defaults to False.
        """
        # log = print if verbose else lambda *args: None
        iterator = tqdm(range(episodios)) if verbose else range(episodios)
        for _ in iterator:
            self.ambiente.reset()
            juego_finalizado = False
            turno_finalizado = False

            # while not juego_finalizado:
            while not juego_finalizado:

                accion = self.elegir_accion()
                estado = self.ambiente.estado()

                q_actual = self.q_table[(estado) + (accion,)]

                recompensa, turno_finalizado, juego_finalizado = self.ambiente.step(
                    accion
                )

                q_siguiente = np.max(self.q_table[self.ambiente.estado()])

                self.q_table[(estado) + (accion,)] = q_actual + self.alpha * (
                    recompensa + self.gamma * q_siguiente - q_actual
                )

    def q_table2pol(self):
        """Convierte la tabla Q en una política."""
        return np.argmax(self.q_table, axis=2)

    def guardar_politica(self, filename: str):
        """Almacena la política del agente en un formato conveniente.

        Args:
            filename (str): Nombre/Path del archivo a generar.
        """

        # Save 0 or 1 in each of the forma[0] * forma[1] cells
        politica = self.q_table2pol()
        np.savetxt("politicas/" + filename, politica, delimiter=",")
