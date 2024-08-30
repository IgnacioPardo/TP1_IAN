import numpy as np
from utils import puntaje_y_no_usados, JUGADA_PLANTARSE, JUGADA_TIRAR, JUGADAS_STR
from collections import defaultdict
from tqdm import tqdm
from jugador import Jugador


class AmbienteDiezMil:
    """
    Ambiente del Juego 10.000.
    """

    def __init__(self):
        """Definir las variables de instancia de un ambiente.
        ¿Qué es propio de un ambiente de 10.000?
        """
        self.turno = 0
        self.estado = EstadoDiezMil()
        self.puntaje_total = 0

    def reset(self):
        """Reinicia el ambiente para volver a realizar un episodio."""
        self.turno = 0
        self.estado = EstadoDiezMil()
        self.puntaje_total = 0

    def calcular_recompensa(self) -> int:
        recompensa = self.estado.puntaje_acumulado_turno
        # Podemos penalizar si self.estado.puntaje_acumulado_turno == 0 por ejemplo
        # O si la cantidad de tiros que me tome es tanto

        return recompensa

    def step(self, accion):
        """Dada una acción devuelve una recompensa.
        El estado es modificado acorde a la acción y su interacción con el ambiente.
        Podría ser útil devolver si terminó o no el turno.

        Args:
            accion: Acción elegida por un agente.

        Returns:
            tuple[int, bool]: Una recompensa y un flag que indica si terminó el turno.
        """

        finalizado: bool = False

        if accion == JUGADA_PLANTARSE:
            self.estado.fin_turno()
            finalizado = True
            recompensa = self.calcular_recompensa()

        elif accion == JUGADA_TIRAR:
            puntaje, dados = puntaje_y_no_usados(np.random.randint(1, 7, 6))
            self.estado.actualizar_estado(puntaje, len(dados))
            recompensa = self.calcular_recompensa()

            if puntaje == 0:
                self.estado.fin_turno()
                finalizado = True

        return recompensa, finalizado


class EstadoDiezMil:
    """Representación de un estado del juego 10.000."""

    def __init__(self):
        """Definir qué hace a un estado de diez mil.
        Recordar que la complejidad del estado repercute en la complejidad de la tabla del agente de q-learning.

        Turnos, tirada por turno

        Estado S = (
            puntaje_acumulado_turno,
            dados_disponibles,
            nro_tirada?,
        )

        """

        self.puntaje_acumulado_turno: int = 0
        self.dados_disponibles: int = 6
        self.turno: int = 0
        self.nro_tirada: int = 0

    def actualizar_estado(self, *args, **kwargs) -> None:
        """Modifica las variables internas del estado luego de una tirada.

        Args:
            args: Puntaje obtenido en la tirada y cantidad de dados disponibles
        """
        if args[0] == 0:
            self.puntaje_acumulado_turno = 0
            self.dados_disponibles = 0
        else:
            self.puntaje_acumulado_turno += args[0]
        self.dados_disponibles = args[1]

    def fin_turno(self):
        """Modifica el estado al terminar el turno."""
        self.puntaje_acumulado_turno = 0
        self.dados_disponibles = 6
        self.turno += 1
        self.nro_tirada: int = 0
        # Faltaria modificar puntaje_acumulado_total cada vez que termina un turno

    def __str__(self):
        """Representación en texto de EstadoDiezMil.
        Ayuda a tener una versión legible del objeto.

        Returns:
            str: Representación en texto de EstadoDiezMil.
        """

        return f"Estado: {self.puntaje_acumulado_turno}, {self.dados_disponibles}, {self.turno}"


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
        self.alpha = alpha  # 0.1
        self.gamma = gamma  # 0.9 ?cuanto me importa el futuro?
        self.epsilon = epsilon  # 0.1
        # Se puede cuantizar los 10k (nunca puedo tener 1,2,3...49, etc puntos)
        forma = (
            20000,
            7,
            2
        )
        q_init_scale = 10000
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

    def elegir_accion(self):
        """
        Selecciona una acción de acuerdo a una política ε-greedy.
        """

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.posibles_acciones)
        else:
            # Select either self.posibles_acciones[0] or self.posibles_acciones[1] based on the Q table
            # if self.q_table[self.ambiente.estado.puntaje_acumulado_turno, self.ambiente.estado.dados_disponibles, 0] > self.q_table[self.ambiente.estado.puntaje_acumulado_turno, self.ambiente.estado.dados_disponibles, 1]:
            #     return self.posibles_acciones[0]
            # else:
            #     return self.posibles_acciones[1]

            return np.argmax(
                self.q_table[
                    self.ambiente.estado.puntaje_acumulado_turno,
                    self.ambiente.estado.dados_disponibles,
                    :,
                ]
            )

    def entrenar(self, episodios: int, verbose: bool = False) -> None:
        """Dada una cantidad de episodios, se repite el ciclo del algoritmo de Q-learning.
        Recomendación: usar tqdm para observar el progreso en los episodios.

        Args:
            episodios (int): Cantidad de episodios a iterar.
            verbose (bool, optional): Flag para hacer visible qué ocurre en cada paso. Defaults to False.
        """
        # log = print if verbose else lambda *args: None
        iter = tqdm(range(episodios)) if verbose else range(episodios)
        for _ in iter:
            self.ambiente.reset()
            finalizado = False

            while not finalizado:
                accion = self.elegir_accion()
                recompensa, finalizado = self.ambiente.step(accion)

                if verbose:
                    iter.set_description(
                        f"Episodio: {_}, Recompensa: {recompensa}, Acción: {JUGADAS_STR[accion]}"
                    )

                q_actual = self.q_table[
                    self.ambiente.estado.puntaje_acumulado_turno,
                    self.ambiente.estado.dados_disponibles,
                    accion,
                ]
                q_siguiente = np.max(
                    self.q_table[
                        self.ambiente.estado.puntaje_acumulado_turno,
                        self.ambiente.estado.dados_disponibles,
                        :,
                    ]
                )
                self.q_table[
                    self.ambiente.estado.puntaje_acumulado_turno,
                    self.ambiente.estado.dados_disponibles,
                    accion,
                ] = q_actual + self.alpha * (
                    recompensa + self.gamma * q_siguiente - q_actual
                )

    def guardar_politica(self, filename: str):
        """Almacena la política del agente en un formato conveniente.

        Args:
            filename (str): Nombre/Path del archivo a generar.
        """

        # Save 0 or 1 in each of the forma[0] * forma[1] cells
        politica = np.argmax(self.q_table, axis=2)
        np.savetxt(filename, politica, delimiter=",")


class JugadorEntrenado(Jugador):
    """Jugador que implementa una política entrenada con un agente de RL.

    Args:
        Jugador (_type_): _description_
    """

    def __init__(self, nombre: str, filename_politica: str):
        self.nombre = nombre
        self.politica = self._leer_politica(filename_politica, SEP=",")
        self.posibles_acciones = [JUGADA_PLANTARSE, JUGADA_TIRAR]

    def _leer_politica(self, filename: str, SEP: str = ","):
        """Carga una politica entrenada con un agente de RL, que está guardada
        en el archivo filename en un formato conveniente.

        Args:
            filename (str): Nombre/Path del archivo que contiene a una política almacenada.
        """
        return np.loadtxt(filename, delimiter=SEP)

    def jugar(
        self,
        puntaje_total: int,
        puntaje_turno: int,
        dados: list[int],
    ) -> tuple[int, list[int]]:
        """Devuelve una jugada y los dados a tirar.

        Args:
            puntaje_total (int): Puntaje total del jugador en la partida.
            puntaje_turno (int): Puntaje en el turno del jugador
            dados (list[int]): Tirada del turno.

        Returns:
            tuple[int,list[int]]: Una jugada y la lista de dados a tirar.
        """

        puntaje, no_usados = puntaje_y_no_usados(dados)
        jugada = self.posibles_acciones[int(self.politica[puntaje_turno, len(no_usados)])]

        if jugada == JUGADA_PLANTARSE:
            return (JUGADA_PLANTARSE, [])
        elif jugada == JUGADA_TIRAR:
            return (JUGADA_TIRAR, no_usados)
