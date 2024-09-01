import numpy as np
from random import randint
from abc import ABC, abstractmethod
from utils import puntaje_y_no_usados, JUGADA_PLANTARSE, JUGADA_TIRAR

class Jugador(ABC):
    @abstractmethod
    def jugar(self, puntaje_total:int, puntaje_turno:int, dados:list[int],
              verbose:bool=False) -> tuple[int,list[int]]:
        pass

class JugadorAleatorio(Jugador):
    def __init__(self, nombre:str):
        self.nombre = nombre
        
    def jugar(self, puntaje_total:int, puntaje_turno:int, dados:list[int],
              verbose:bool=False) -> tuple[int,list[int]]:
        (puntaje, no_usados) = puntaje_y_no_usados(dados)
        if randint(0, 1)==0:
            return (JUGADA_PLANTARSE, [])
        else:
            return (JUGADA_TIRAR, no_usados)

class JugadorSiempreSePlanta(Jugador):
    def __init__(self, nombre:str):
        self.nombre = nombre
        
    def jugar(self, puntaje_total:int, puntaje_turno:int, dados:list[int], 
              verbose:bool=False) -> tuple[int,list[int]]:
        return (JUGADA_PLANTARSE, [])
    
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
        verbose:bool=False
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
        verbose:bool=False
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

