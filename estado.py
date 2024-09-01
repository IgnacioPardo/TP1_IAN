from utils import snap_puntos

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
        # self.turno: int = 0
        # self.nro_tirada: int = 0

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
        # self.turno += 1
        # self.nro_tirada: int = 0
        # Faltaria modificar puntaje_acumulado_total cada vez que termina un turno

    def __str__(self):
        """Representación en texto de EstadoDiezMil.
        Ayuda a tener una versión legible del objeto.

        Returns:
            str: Representación en texto de EstadoDiezMil.
        """

        return f"Estado: {self.puntaje_acumulado_turno}, {self.dados_disponibles}"
    
class EstadoDiezMilv2:
    """Representación de un estado del juego 10.000."""

    def __init__(self):
        """Definir qué hace a un estado de diez mil.
        Recordar que la complejidad del estado repercute en la complejidad de la tabla del agente de q-learning.

        Estado S = (
            puntaje_acumulado_turno (11 posibles valores: 1-> 0 2->[1, 1000), ..., 11->[9000, 10000)),
            dados_disponibles,
        )

        """

        self.puntaje_acumulado_turno_snap: int = 0
        self.dados_disponibles: int = 6
        self.nro_tirada: int = 0

    def actualizar_estado(self, *args, **kwargs) -> None:
        """Modifica las variables internas del estado luego de una tirada.

        Args:
            args: (Puntaje obtenido en la tirada, cantidad de dados disponibles)
        """
        if args[0] == 0:
            self.puntaje_acumulado_turno_snap = 0
            self.dados_disponibles = 0
        else:
            self.puntaje_acumulado_turno_snap = snap_puntos(args[0])
            self.dados_disponibles = args[1]
        self.nro_tirada += 1

    def fin_turno(self):
        """Modifica el estado al terminar el turno."""
        self.puntaje_acumulado_turno_snap = 0
        self.dados_disponibles = 6
        self.nro_tirada: int = 0

    def __str__(self):
        """Representación en texto de EstadoDiezMil.
        Ayuda a tener una versión legible del objeto.

        Returns:
            str: Representación en texto de EstadoDiezMil.
        """

        return f"Estado: {self.puntaje_acumulado_turno_snap}, {self.dados_disponibles}"