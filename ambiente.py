import numpy as np
from utils import JUGADA_PLANTARSE, JUGADA_TIRAR, puntaje_y_no_usados, RangeSnaper
from estado import EstadoDiezMilv2


class AmbienteDiezMil:
    """
    Ambiente del Juego 10.000.
    """

    def __init__(
        self,
        multiplicador_recompensa: int = 1,
        max_turnos: int = 40,
        rs: RangeSnaper = RangeSnaper(),
    ):
        """Define instance variables for a 10,000-point game environment."""
        self.range_snaper = rs
        self.estado = EstadoDiezMilv2(rs)
        self.puntaje_total = 0
        self.turno = 0
        self.multiplicador_recompensa = multiplicador_recompensa
        self.max_turnos = max_turnos

    def reset(self):
        """Reinicia el ambiente para volver a realizar un episodio."""
        self.estado = EstadoDiezMilv2(self.range_snaper)
        self.puntaje_total = 0
        self.turno = 0

    def step(self, accion):
        """Dada una acción devuelve una recompensa.
        El estado es modificado acorde a la acción y su interacción con el ambiente.
        Podría ser útil devolver si terminó o no el turno.

        Args:
            accion: Acción elegida por un agente.

        Returns:
            tuple[int, bool,  bool]: Una recompensa, un flag que indica si terminó el turno y un flag que indica si termino el juego.
        """

        turno_finalizado: bool = False
        juego_finalizado: bool = False

        if accion == JUGADA_PLANTARSE:
            turno_finalizado = True
            self.puntaje_total += self.estado.puntaje_turno
            # recompensa = self.estado.puntaje_turno_miles * self.multiplicador_recompensa / self.turno
            recompensa = self.estado.puntaje_turno * self.multiplicador_recompensa

        elif accion == JUGADA_TIRAR:
            puntaje, no_usados = puntaje_y_no_usados(
                np.random.randint(1, 7, self.estado.dados_disponibles)
            )
            if puntaje == 0:
                turno_finalizado = True
                # recompensa = -self.estado.puntaje_turno_miles * self.multiplicador_recompensa / self.turno
                recompensa = 0
            else:
                self.estado.actualizar_estado(puntaje, len(no_usados))
                # recompensa = self.estado.puntaje_turno_miles * self.multiplicador_recompensa / self.turno
                recompensa = self.estado.puntaje_turno * self.multiplicador_recompensa

        if (self.puntaje_total) >= 10000:
            self.puntaje_total = 10000
            turno_finalizado = True
            juego_finalizado = True

        if self.turno >= self.max_turnos:
            juego_finalizado = True
            turno_finalizado = True

        if turno_finalizado:
            self.turno += 1
            self.estado.fin_turno()

        # recompensa = 0

        return recompensa, turno_finalizado, juego_finalizado
