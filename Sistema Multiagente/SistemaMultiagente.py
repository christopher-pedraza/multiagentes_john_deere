import tracemalloc
import asyncio
import websockets
import agentpy as ap
import numpy as np
import json

# Inicia la herramienta de tracemalloc para el seguimiento de la asignación de memoria
tracemalloc.start()


# Función para normalizar un vector
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# Clase que representa una celda en el campo
class CeldaCampo(ap.Agent):
    def setup(self):
        self.isCosechado = False
        self.densidad = self.p.densidad
        self.identificador = "celda"

    def cosechar(self):
        if not self.isCosechado:
            self.isCosechado = True
            return self.densidad
        else:
            return 0


# Clase que representa una cosechadora en el campo
class Cosechadora(ap.Agent):
    def setup(self):
        self.velocity = 1
        self.identificador = "cosechadora"
        self.capacidad_max = self.p.capacidad_max
        self.capacidad = 0
        self.estado = "cosechando"

    def setup_pos(self, space):
        self.space = space
        self.neighbors = space.neighbors
        self.pos = space.positions[self]

    def update_velocity(self):
        if self.estado == "cosechando":
            pos = self.pos
            ndim = self.p.ndim

            # Regla 2 - Separación
            v2 = np.zeros(ndim)
            for nb in self.neighbors(self, distance=self.p.inner_radius):
                if nb.identificador != "celda":
                    v2 -= nb.pos - pos
            v2 *= self.p.separation_strength

            # Regla 4 - Bordes
            v4 = np.zeros(ndim)
            d = self.p.border_distance
            s = self.p.border_strength
            for i in range(ndim):
                if pos[i] < d:
                    v4[i] += s
                elif pos[i] > self.space.shape[i] - d:
                    v4[i] -= s

            self.velocity += v2 + v4
            self.velocity = normalize(self.velocity)

    def update_position(self):
        if self.estado == "cosechando":
            self.space.move_by(self, self.velocity)

    def cosechar(self):
        for nb in self.neighbors(self, distance=self.p.harvest_radius):
            if nb.identificador == "celda" and not nb.isCosechado:
                self.capacidad += nb.cosechar()

                if self.capacidad + self.p.densidad > self.capacidad_max:
                    self.velocity = 0
                    self.estado = "lleno"


# Clase que representa un tractor en el campo
class Tractor(ap.Agent):
    def setup(self):
        self.velocity = 1.5
        self.identificador = "tractor"
        self.target_position = np.array([10, 10])
        self.moving = False

    def move(self):
        if self.moving:
            # Calcular el vector de dirección hacia la posición objetivo
            direccion = self.target_position - self.pos
            distancia = np.linalg.norm(direccion)
            direccion = normalize(direccion)

            # Moverse hacia la posición objetivo
            if distancia > self.velocity:
                self.space.move_by(self, direccion * self.velocity)
            else:
                # Si está cerca del objetivo, ajustar a la posición objetivo y dejar de moverse
                self.moving = False
                for nb in self.neighbors(self, distance=self.p.tractor_radius):
                    if nb.identificador == "cosechadora" and nb.estado == "esperando":
                        nb.capacidad = 0
                        nb.estado = "cosechando"
                        nb.velocity = 1
                        nb.update_velocity()
                        nb.update_position()
                        self.target_position = np.array([0, 0])
                        self.moving = True

        else:
            for nb in self.neighbors(self, distance=self.p.dimensiones_campo**2):
                if nb.identificador == "cosechadora" and nb.estado == "lleno":
                    self.target_position = nb.pos
                    self.moving = True
                    nb.estado = "esperando"
                    break

    def setup_pos(self, space):
        self.space = space
        self.neighbors = space.neighbors
        self.pos = space.positions[self]

    def update_velocity(self):
        pos = self.pos
        ndim = self.p.ndim

        # Regla 2 - Separación
        v2 = np.zeros(ndim)
        for nb in self.neighbors(self, distance=self.p.inner_radius):
            if nb.identificador != "celda":
                v2 -= nb.pos - pos
        v2 *= self.p.separation_strength

        # Regla 4 - Bordes
        v4 = np.zeros(ndim)
        d = self.p.border_distance
        s = self.p.border_strength
        for i in range(ndim):
            if pos[i] < d:
                v4[i] += s
            elif pos[i] > self.space.shape[i] - d:
                v4[i] -= s

        # Actualizar la velocidad
        self.velocity += v2 + v4
        self.velocity = normalize(self.velocity)

    def update_position(self):
        self.space.move_by(self, self.velocity)


# Clase principal que representa el modelo del campo
class FieldModel(ap.Model):
    def setup(self):
        self.space = ap.Space(self, shape=[self.p.size] * self.p.ndim)
        self.cosechadoras = ap.AgentList(
            self, self.p.cosechadora_population, Cosechadora
        )
        self.tractors = ap.AgentList(self, self.p.tractor_population, Tractor)
        self.space.add_agents(self.cosechadoras, random=True)
        self.space.add_agents(self.tractors, random=True)
        self.cosechadoras.setup_pos(self.space)
        self.tractors.setup_pos(self.space)

        # Crear celdas_campo sin agregarlas al espacio aún
        self.celdas_campo = ap.AgentList(
            self, self.p.dimensiones_campo**2, CeldaCampo
        )

        # Asignar manualmente posiciones a cada celda_campo
        grid_size = self.p.dimensiones_campo
        for x in range(grid_size):
            for y in range(grid_size):
                index = x * grid_size + y
                celda = self.celdas_campo[index]
                # celda.setup_pos(self.space)
                self.space.positions[celda] = (x, y)

    # Envia las posiciones de los agentes a través de WebSocket
    async def send_positions(self, websocket):
        positions_cosechadoras = {
            f"Cosechadora_{str(agent)}": agent.pos.tolist()
            for agent in self.cosechadoras
        }
        capacidad_cosechadoras = {
            f"Capacidad_{str(agent)}": [float(agent.capacidad)]
            for agent in self.cosechadoras
        }
        positions_tractors = {
            f"Tractor_{str(agent)}": agent.pos.tolist() for agent in self.tractors
        }
        positions = {
            **positions_cosechadoras,
            **capacidad_cosechadoras,
            **positions_tractors,
        }
        await websocket.send(json.dumps(positions))

    # Manejador de WebSocket para la simulación
    async def ws_handler_inner(self, websocket):
        try:
            while True:
                # Actualizar posiciones de agentes existentes
                self.cosechadoras.update_velocity()  # Ajustar dirección
                self.cosechadoras.update_position()  # Moverse en la nueva dirección
                self.cosechadoras.cosechar()
                self.tractors.move()
                await self.send_positions(websocket)
                await asyncio.sleep(0.1)  # Ajustar la frecuencia según sea necesario
        except websockets.exceptions.ConnectionClosed:
            pass

    # Manejador principal de WebSocket
    async def ws_handler(self, websocket, path):
        await self.ws_handler_inner(websocket)

    # Ejecuta la simulación y WebSocket
    async def run_simulation_with_websocket(self):
        self.space = ap.Space(self, shape=[self.p.size] * self.p.ndim)
        self.cosechadoras = ap.AgentList(
            self, self.p.cosechadora_population, Cosechadora
        )
        self.tractors = ap.AgentList(self, self.p.tractor_population, Tractor)
        self.space.add_agents(self.cosechadoras, random=True)
        self.cosechadoras.setup_pos(self.space)

        # Crear celdas_campo sin agregarlas al espacio aún
        self.celdas_campo = ap.AgentList(
            self, self.p.dimensiones_campo**2, CeldaCampo
        )

        for i in range(self.p.tractor_population):
            self.space.positions[self.tractors[i]] = np.ndarray(
                (2,), buffer=np.array([0, 0]), dtype=int
            )
        self.tractors.setup_pos(self.space)

        # Asignar manualmente posiciones a cada celda_campo
        grid_size = self.p.dimensiones_campo
        for x in range(grid_size):
            for y in range(grid_size):
                index = x * grid_size + y
                celda = self.celdas_campo[index]
                # celda.setup_pos(self.space)
                self.space.positions[celda] = (x, y)

        loop = asyncio.get_running_loop()

        # Habilitar tracemalloc dentro del bucle de eventos
        tracemalloc.start()

        # Inicia el servidor WebSocket
        server = await websockets.serve(
            lambda ws, path: self.ws_handler(ws, path),
            "localhost",
            8765,
        )

        try:
            for _ in range(self.p.steps):
                self.step()
                await asyncio.sleep(0.1)
        finally:
            # Cierre del servidor WebSocket
            server.close()
            await server.wait_closed()


# Parámetros del modelo en 2D
parameters2D = {
    "size": 50,
    "seed": 123,
    "steps": 1000,
    "ndim": 2,
    "dimensiones_campo": 50,
    "densidad": 10,
    "capacidad_max": 1000,
    "cosechadora_population": 5,
    "tractor_population": 2,
    "inner_radius": 5,  # 3
    "outer_radius": 10,  # 10
    "harvest_radius": 1,
    "border_distance": 3,  # 10
    "tractor_radius": 3,
    "cohesion_strength": 0.005,
    "separation_strength": 0.1,
    "alignment_strength": 0.3,
    "border_strength": 0.5,
}

# Crea una instancia del modelo y ejecuta la simulación con WebSocket
model = FieldModel(parameters2D)
asyncio.run(model.run_simulation_with_websocket())
