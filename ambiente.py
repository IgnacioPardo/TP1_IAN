import numpy as np
from utils import JUGADA_PLANTARSE, JUGADA_TIRAR, puntaje_y_no_usados
from estado import EstadoDiezMilv2


class AmbienteDiezMil:
    """
    Ambiente del Juego 10.000.
    """

    def __init__(self):
        """Define instance variables for a 10,000-point game environment."""
        self.turno = 0
        self.estado = EstadoDiezMilv2()
        self.puntaje_total = 0
        self.puntaje_turno = 0
        self.dados_disponibles = 6

    def reset(self):
        """Reinicia el ambiente para volver a realizar un episodio."""
        self.turno = 0
        self.estado = EstadoDiezMilv2()
        self.puntaje_total = 0
        self.puntaje_turno = 0
        self.dados_disponibles = 6
    
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
            self.puntaje_total += self.puntaje_turno

        elif accion == JUGADA_TIRAR:
            puntaje, dados = puntaje_y_no_usados(np.random.randint(1, 7, self.dados_disponibles))

            if puntaje == 0:
                turno_finalizado = True
                self.puntaje_turno = 0
            else:
                self.puntaje_turno += puntaje
                self.dados_disponibles = len(dados)

        recompensa = self.puntaje_turno
            
        if (self.puntaje_total + self.puntaje_turno) >= 10000:
            self.puntaje_total = 10000
            turno_finalizado = True
            juego_finalizado = True

        if turno_finalizado:
            self.estado.fin_turno()
            self.puntaje_turno = 0
            self.dados_disponibles = 6
            self.turno += 1
        else:
            self.estado.actualizar_estado(self.puntaje_turno, self.dados_disponibles)

        return recompensa, turno_finalizado, juego_finalizado