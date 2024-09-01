import numpy as np
from utils import JUGADA_PLANTARSE, JUGADA_TIRAR, puntaje_y_no_usados
from estado import EstadoDiezMilv2


class AmbienteDiezMil:
    """
    Ambiente del Juego 10.000.
    """

    def __init__(self, reward_func_type: int = 0):
        """Define instance variables for a 10,000-point game environment."""
        self.turno = 0
        self.estado = EstadoDiezMilv2()
        self.puntaje_total = 0
        self.puntaje_turno = 0

        self.calcular_recompensa = [
            self.calcular_recompensa_0,
        ][reward_func_type]

    def calcular_recompensa_0(self, *args, **kwargs):
        """Calcula la recompensa de un turno."""
        if self.puntaje_turno == 0:
            return -1.5 * (self.puntaje_total // 1000)
        elif self.puntaje_turno + self.puntaje_total >= 10000:
            return 100 / (self.turno + 1)
        else:
            return self.puntaje_turno / 1000
    
    def reset(self):
        """Reinicia el ambiente para volver a realizar un episodio."""
        self.turno = 0
        self.estado = EstadoDiezMilv2()
        self.puntaje_total = 0
        self.puntaje_turno = 0

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
            recompensa = self.calcular_recompensa()
            self.estado.fin_turno()
            self.puntaje_turno = 0

        elif accion == JUGADA_TIRAR:
            puntaje, dados = puntaje_y_no_usados(np.random.randint(1, 7, self.estado.dados_disponibles))
            self.estado.actualizar_estado(puntaje, len(dados))

            if puntaje == 0:
                turno_finalizado = True
                self.puntaje_turno = 0
                recompensa = self.calcular_recompensa()
                self.estado.fin_turno()
            else:
                self.puntaje_turno += puntaje
                recompensa = self.calcular_recompensa()
                self.puntaje_total += puntaje

        if (self.puntaje_total + self.puntaje_turno) >= 10000:
            self.puntaje_total = 10000
            self.puntaje_turno = 0
            self.estado.fin_turno()
            turno_finalizado = True
            juego_finalizado = True

        return recompensa, turno_finalizado, juego_finalizado