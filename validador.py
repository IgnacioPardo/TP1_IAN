import numpy as np
from diezmil import JuegoDiezMil
from jugador import Jugador, JugadorFromPolicy
from ambiente import AmbienteDiezMil
from utils import JUGADA_PLANTARSE, JUGADA_TIRAR


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
            cantidad_turnos, _ = juego.jugar(verbose=False)
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
        jugador = JugadorFromPolicy(politica, self.ambiente.range_snaper)
        return self.validar_jugador(jugador, episodios)