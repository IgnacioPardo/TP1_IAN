from typing import Callable
import numpy as np
from diezmil import JuegoDiezMil
from utils import puntaje_y_no_usados, JUGADA_PLANTARSE, JUGADA_TIRAR, JUGADAS_STR
from collections import defaultdict
from tqdm import tqdm
from jugador import Jugador


class AmbienteDiezMil:
    """
    Ambiente del Juego 10.000.
    """

    def __init__(self, reward_func_type: int = 0):
        """Define instance variables for a 10,000-point game environment."""
        self.turno = 0
        self.estado = EstadoDiezMil()
        self.puntaje_total = 0
        self.factor_aleatoriedad = 0.05
        self.limite_superior = 10000
        self.limite_inferior = 5000

        self.calcular_recompensa = [
            self.calcular_recompensa_0,
            self.calcular_recompensa_1,
            self.calcular_recompensa_2,
            self.calcular_recompensa_3,
            self.calcular_recompensa_4,
            self.calcular_recompensa_5,
            self.calcular_recompensa_6,
            self.calcular_recompensa_7,
            self.calcular_recompensa_8,
            self.calcular_recompensa_9,
            self.calcular_recompensa_10,
            self.calcular_recompensa_11,
            self.calcular_recompensa_12,
            self.calcular_recompensa_13,
        ][reward_func_type]

    def calcular_recompensa_0(self):
        return self.estado.puntaje_acumulado_turno

    def calcular_recompensa_1(self):
        return self.estado.puntaje_acumulado_turno / (self.turno + 1)

    def calcular_recompensa_2(self):
        return self.estado.puntaje_acumulado_turno / (self.turno + 1) ** 2

    def calcular_recompensa_3(self):
        return 1 if self.puntaje_total >= 10000 else 0

    def calcular_recompensa_4(self):
        return self.estado.puntaje_acumulado_turno / 10000

    def calcular_recompensa_5(self):
        return (self.estado.puntaje_acumulado_turno + self.puntaje_total) / (
            self.turno + 2
        )

    def calcular_recompensa_6(self):
        return self.estado.puntaje_acumulado_turno * self.turno

    def calcular_recompensa_7(self):
        return self.estado.puntaje_acumulado_turno / max(1, self.puntaje_total)

    def calcular_recompensa_8(self):
        return self.estado.puntaje_acumulado_turno * (self.factor_aleatoriedad + 1)

    def calcular_recompensa_9(self):
        return (
            self.estado.puntaje_acumulado_turno / (self.turno + 1)
            if self.puntaje_total < 5000
            else self.estado.puntaje_acumulado_turno * 2
        )

    def calcular_recompensa_10(self):
        return (self.estado.puntaje_acumulado_turno / (self.turno + 1)) ** (
            self.puntaje_total / 10000
        )

    def calcular_recompensa_11(self):
        return (
            self.estado.puntaje_acumulado_turno * (1 + self.factor_aleatoriedad)
            if self.puntaje_total > self.limite_superior
            else self.estado.puntaje_acumulado_turno
        )

    def calcular_recompensa_12(self):
        return self.estado.puntaje_acumulado_turno / (
            1 + abs(self.puntaje_total - 10000)
        )

    def calcular_recompensa_13(self):
        return (
            self.estado.puntaje_acumulado_turno + self.turno * self.factor_aleatoriedad
        ) / max(1, self.turno)

    def reset(self):
        """Reinicia el ambiente para volver a realizar un episodio."""
        self.turno = 0
        self.estado = EstadoDiezMil()
        self.puntaje_total = 0

    def step(self, accion):
        """Dada una acción devuelve una recompensa.
        El estado es modificado acorde a la acción y su interacción con el ambiente.
        Podría ser útil devolver si terminó o no el turno.

        Args:
            accion: Acción elegida por un agente.

        Returns:
            tuple[int, bool]: Una recompensa y un flag que indica si terminó el turno.
        """

        if self.puntaje_total >= 10000:
            self.estado.fin_turno()
            recompensa = 10000
            finalizado = True
            return recompensa, finalizado

        finalizado: bool = False

        if accion == JUGADA_PLANTARSE:
            self.estado.fin_turno()
            finalizado = True
            recompensa = self.calcular_recompensa()

        elif accion == JUGADA_TIRAR:
            puntaje, dados = puntaje_y_no_usados(np.random.randint(1, 7, 6))
            self.estado.actualizar_estado(puntaje, len(dados))
            self.puntaje_total += puntaje
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


class Validador:
    """Clase que permite validar una política entrenada."""

    def __init__(self, ambiente: AmbienteDiezMil):
        """Definir las variables internas de un Validador.

        Args:
            ambiente (AmbienteDiezMil): Ambiente con el que interactuará el agente.
        """
        self.ambiente = ambiente
        self.posibles_acciones = [JUGADA_PLANTARSE, JUGADA_TIRAR]

    def validar_jugador(self, jugador: Jugador, episodios: int) -> float:
        """Dado un jugador, se valida su desempeño en el ambiente.

        Args:
            jugador (Jugador): Jugador a validar.
            episodios (int): Cantidad de episodios a iterar.

        Returns:
            float: Turnos promedio necesarios para llegar a 10.000.
        """
        turnos = []

        for _ in range(episodios):
            juego = JuegoDiezMil(jugador)
            cantidad_turnos, puntaje_final = juego.jugar(verbose=False)
            turnos.append(cantidad_turnos)

        return np.mean(turnos)

    def validar_politica(self, politica: np.array, episodios: int) -> float:
        """Dada una política entrenada, se valida su desempeño en el ambiente.

        Args:
            politica (np.array): Política entrenada.
            episodios (int): Cantidad de episodios a iterar.

        Returns:
            float: Turnos promedio necesarios para llegar a 10.000.
        """
        jugador = JugadorFromPolicy(politica)
        return self.validar_jugador(jugador, episodios)


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
        # self.q_table = np.random.uniform(0, 1, shape) * q_init_scale
        self.q_table = np.zeros(shape)

    def elegir_accion(self):
        """
        Selecciona una acción de acuerdo a una política ε-greedy.
        """

        if self.ambiente.estado.dados_disponibles == 0:
            return JUGADA_PLANTARSE

        if self.ambiente.estado.puntaje_acumulado_turno >= 10000:
            return JUGADA_PLANTARSE

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.posibles_acciones)
        else:
            # Select either self.posibles_acciones[0] or self.posibles_acciones[1] based on the Q table
            # if self.q_table[self.ambiente.estado.puntaje_acumulado_turno, self.ambiente.estado.dados_disponibles, 0] > self.q_table[self.ambiente.estado.puntaje_acumulado_turno, self.ambiente.estado.dados_disponibles, 1]:
            #     return self.posibles_acciones[0]
            # else:
            #     return self.posibles_acciones[1]

            pts = np.clip(self.ambiente.estado.puntaje_acumulado_turno, 0, 10000)
            return self.posibles_acciones[
                int(
                    np.argmax(
                        self.q_table[
                            pts,
                            self.ambiente.estado.dados_disponibles,
                            :,
                        ]
                    )
                )
            ]

    def entrenar(self, episodios: int, verbose: bool = False, validate: bool = False) -> None:
        """Dada una cantidad de episodios, se repite el ciclo del algoritmo de Q-learning.
        Recomendación: usar tqdm para observar el progreso en los episodios.

        Args:
            episodios (int): Cantidad de episodios a iterar.
            verbose (bool, optional): Flag para hacer visible qué ocurre en cada paso. Defaults to False.
        """
        # log = print if verbose else lambda *args: None
        iter = tqdm(range(episodios)) if verbose else range(episodios)
        for ep in iter:
            self.ambiente.reset()
            finalizado = False

            while not finalizado:

                accion = self.elegir_accion()
                pts = self.ambiente.estado.puntaje_acumulado_turno
                dados = self.ambiente.estado.dados_disponibles

                q_actual = self.q_table[
                    pts,
                    dados,
                    accion,
                
                ]

                recompensa, finalizado = self.ambiente.step(accion)

                q_siguiente = np.max(
                    self.q_table[
                        self.ambiente.estado.puntaje_acumulado_turno,
                        self.ambiente.estado.dados_disponibles,
                        :,
                    ]
                )

                self.q_table[
                    pts,
                    dados,
                    accion,
                ] = q_actual + self.alpha * (
                    recompensa + self.gamma * q_siguiente - q_actual
                )

            val_promedio = 0
            if validate:
                val = Validador(self.ambiente)
                val_promedio = val.validar_politica(self.q_table2pol(), 100)

            if verbose:
                iter.set_description(
                    f"Episodio {ep}" + f" - Validación: {val_promedio}" if validate else ""
                )
            yield val_promedio

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
        pts = puntaje_turno  # + puntaje
        if puntaje == 0:
            pts = 0

        jugada = self.posibles_acciones[
            int(self.politica[pts, len(no_usados)])
        ]

        if jugada == JUGADA_PLANTARSE:
            return (JUGADA_PLANTARSE, [])
        elif jugada == JUGADA_TIRAR:
            return (JUGADA_TIRAR, no_usados)

        # puntaje, no_usados = puntaje_y_no_usados(dados)
        # COMPLETAR
        # estado = ...
        # jugada = self.politica[estado]

        # if jugada==JUGADA_PLANTARSE:
        #     return (JUGADA_PLANTARSE, [])
        # elif jugada==JUGADA_TIRAR:
        #     return (JUGADA_TIRAR, no_usados)


class JugadorFromPolicy(Jugador):
    """Jugador que implementa una política entrenada con un agente de RL.

    Args:
        Jugador (_type_): _description_
    """

    def __init__(self, politica: np.array):
        self.politica = politica
        self.posibles_acciones = [JUGADA_PLANTARSE, JUGADA_TIRAR]

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
        pts = puntaje_turno  # + puntaje
        if puntaje == 0:
            pts = 0

        jugada = self.posibles_acciones[
            int(self.politica[pts, len(no_usados)])
        ]

        if jugada == JUGADA_PLANTARSE:
            return (JUGADA_PLANTARSE, [])
        elif jugada == JUGADA_TIRAR:
            return (JUGADA_TIRAR, no_usados)
