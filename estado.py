from utils import RangeSnaper


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

    def __init__(self, rs: RangeSnaper):
        """Definir qué hace a un estado de diez mil.
        Recordar que la complejidad del estado repercute en la complejidad de la tabla del agente de q-learning.

        Estado S = (
            puntaje_acumulado_turno (12 posibles valores: 1-> 0 2->[1, 1000), ..., 11->[9000, 10000)),
            dados_disponibles (7 posibles valores: 0-> 0, 1-> 1, ..., 6-> 6)
        )

        """
        # Solo para calcular el puntaje en miles
        # NO es parte del estado
        self.puntaje_turno: int = 0
        self.rs = rs

        # Variables del estado
        self.puntaje_turno_miles: int = 0
        self.dados_disponibles: int = 6

    def actualizar_estado(self, puntaje, dados_disponibles) -> None:
        """Modifica las variables internas del estado luego de una tirada.

        Args:
            args: (Puntaje acumulado del turno, cantidad de dados disponibles)
        """
        self.puntaje_turno += puntaje
        self.puntaje_turno_miles = self.rs.snap(self.puntaje_turno)
        self.dados_disponibles = dados_disponibles

    def fin_turno(self):
        """Modifica el estado al terminar el turno."""
        self.puntaje_turno_miles = 0
        self.puntaje_turno = 0
        self.dados_disponibles = 6

    def __call__(self):
        return self.puntaje_turno_miles, self.dados_disponibles

    def __str__(self):
        """Representación en texto de EstadoDiezMil.
        Ayuda a tener una versión legible del objeto.

        Returns:
            str: Representación en texto de EstadoDiezMil.
        """

        return f"Estado: {self.puntaje_turno_miles}, {self.dados_disponibles}"
