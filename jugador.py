import numpy as np
from random import randint
from abc import ABC, abstractmethod
from utils import puntaje_y_no_usados, JUGADA_PLANTARSE, JUGADA_TIRAR, RangeSnaper


class Jugador(ABC):
    @abstractmethod
    def jugar(
        self,
        puntaje_total: int,
        puntaje_turno: int,
        dados: list[int],
        verbose: bool = False,
    ) -> tuple[int, list[int]]:
        pass


class JugadorAleatorio(Jugador):
    def __init__(self, nombre: str):
        self.nombre = nombre

    def jugar(
        self,
        puntaje_total: int,
        puntaje_turno: int,
        dados: list[int],
        verbose: bool = False,
    ) -> tuple[int, list[int]]:
        (puntaje, no_usados) = puntaje_y_no_usados(dados)
        if randint(0, 1) == 0:
            return (JUGADA_PLANTARSE, [])
        else:
            return (JUGADA_TIRAR, no_usados)


class JugadorSiempreSePlanta(Jugador):
    def __init__(self, nombre: str):
        self.nombre = nombre

    def jugar(
        self,
        puntaje_total: int,
        puntaje_turno: int,
        dados: list[int],
        verbose: bool = False,
    ) -> tuple[int, list[int]]:
        return (JUGADA_PLANTARSE, [])


class JugadorEntrenado(Jugador):
    """Jugador que implementa una política entrenada con un agente de RL.

    Args:
        Jugador (_type_): _description_
    """

    def __init__(self, nombre: str, filename_politica: str, rs: RangeSnaper):
        self.nombre = nombre
        self.politica = self._leer_politica(filename_politica, SEP=",")
        self.posibles_acciones = [JUGADA_PLANTARSE, JUGADA_TIRAR]
        self.range_snaper = rs

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
        verbose: bool = False,
    ) -> tuple[int, list[int]]:
        """Devuelve una jugada y los dados a tirar.

        Args:
            puntaje_total (int): Puntaje total del jugador en la partida.
            puntaje_turno (int): Puntaje en el turno del jugador
            dados (list[int]): Tirada del turno.

        Returns:
            tuple[int,list[int]]: Una jugada y la lista de dados a tirar.
        """

        _, no_usados = puntaje_y_no_usados(dados)
        estado = (self.range_snaper.snap(puntaje_turno), len(no_usados))

        jugada = self.posibles_acciones[int(self.politica[estado])]

        if jugada == JUGADA_PLANTARSE:
            return (JUGADA_PLANTARSE, [])
        elif jugada == JUGADA_TIRAR:
            return (JUGADA_TIRAR, no_usados)


class JugadorFromPolicy(Jugador):
    """Jugador que implementa una política entrenada con un agente de RL.

    Args:
        Jugador (_type_): _description_
    """

    def __init__(self, politica: np.array, rs: RangeSnaper):
        self.politica = politica
        self.posibles_acciones = [JUGADA_PLANTARSE, JUGADA_TIRAR]
        self.range_snaper = rs

    def jugar(
        self,
        puntaje_total: int,
        puntaje_turno: int,
        dados: list[int],
        verbose: bool = False,
    ) -> tuple[int, list[int]]:
        """Devuelve una jugada y los dados a tirar.

        Args:
            puntaje_total (int): Puntaje total del jugador en la partida.
            puntaje_turno (int): Puntaje en el turno del jugador
            dados (list[int]): Tirada del turno.

        Returns:
            tuple[int,list[int]]: Una jugada y la lista de dados a tirar.
        """

        _, no_usados = puntaje_y_no_usados(dados)
        estado = (self.range_snaper.snap(puntaje_turno), len(no_usados))

        jugada = self.posibles_acciones[int(self.politica[estado])]

        if jugada == JUGADA_PLANTARSE:
            return (JUGADA_PLANTARSE, [])
        elif jugada == JUGADA_TIRAR:
            return (JUGADA_TIRAR, no_usados)


# from DQN import DQNetwork
# import torch
# class JugadorFromDQNPolicy(Jugador):
#     """Jugador que implementa una política entrenada con un agente DQN.

#     Args:
#         Jugador (_type_): Clase base del jugador.
#     """

#     def __init__(self, model_path: str):
#         """Inicializa el jugador cargando la red neuronal entrenada.

#         Args:
#             model_path (str): Ruta al archivo con los pesos de la red neuronal entrenada.
#         """
#         self.posibles_acciones = [JUGADA_PLANTARSE, JUGADA_TIRAR]
#         input_dim = 2  # PUNTOS TURNO + DADOS DISPONIBLES
#         output_dim = len(self.posibles_acciones)
#         self.q_network = DQNetwork(input_dim, output_dim)
#         self.q_network.load_state_dict(torch.load('politicas/' + model_path))
#         self.q_network.eval()  # Cambiar a modo evaluación

#     def jugar(
#         self,
#         puntaje_total: int,
#         puntaje_turno: int,
#         dados: list[int],
#         verbose:bool=False
#     ) -> tuple[int, list[int]]:
#         """Devuelve una jugada y los dados a tirar.

#         Args:
#             puntaje_total (int): Puntaje total del jugador en la partida.
#             puntaje_turno (int): Puntaje en el turno del jugador.
#             dados (list[int]): Tirada del turno.

#         Returns:
#             tuple[int, list[int]]: Una jugada y la lista de dados a tirar.
#         """
#         _, no_usados = puntaje_y_no_usados(dados)
#         estado = (snap_puntos(puntaje_turno), len(no_usados))

#         # Convertir el estado a un tensor para la red neuronal
#         estado_tensor = torch.tensor(estado, dtype=torch.float32)

#         # Predecir la acción con la red neuronal
#         with torch.no_grad():
#             q_values = self.q_network(estado_tensor)
#         accion_idx = torch.argmax(q_values).item()
#         jugada = self.posibles_acciones[accion_idx]

#         if jugada == JUGADA_PLANTARSE:
#             return (JUGADA_PLANTARSE, [])
#         elif jugada == JUGADA_TIRAR:
#             return (JUGADA_TIRAR, no_usados)
