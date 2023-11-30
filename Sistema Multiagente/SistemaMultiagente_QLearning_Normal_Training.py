import tracemalloc
import asyncio
import websockets
import agentpy as ap
import numpy as np
import json
from copy import deepcopy

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
        self.pertenencia = None

    def setup_pertenencia(self, id_cosechadora):
        self.pertenencia = id_cosechadora

    def cosechar(self):
        if not self.isCosechado:
            self.isCosechado = True
            return self.densidad
        else:
            return 0


# Clase que representa una cosechadora en el campo
class Cosechadora(ap.Agent):
    def setup(self):
        self.velocity = 1.0
        self.identificador = "cosechadora"
        self.capacidad_max = self.p.capacidad_max
        self.capacidad = 0
        self.estado = "cosechando"  # cosechando / lleno / esperando
        self.moving = False
        # -1: None
        # 0: Up
        # 1: Down
        # 2: Left
        # 3: Right
        # 4: Complete
        self.current_direction = 0
        # -1: None
        # 0: Up
        # 1: Down
        # 2: Left
        # 3: Right
        # 4: Complete
        self.previous_rotation = -1

    def move(self, direction):
        reward = 0
        done = False

        if self.moving:
            # Calcular el vector de dirección hacia la posición objetivo
            direccion = self.target_position - self.pos
            distancia = np.linalg.norm(direccion)
            direccion = normalize(direccion)

            # Moverse hacia la posición objetivo
            if distancia >= self.velocity:
                # self.space.move_to(self, self.target_position)
                self.space.move_by(self, direccion * self.velocity)

            else:
                if (
                    self.target_position[0] > self.p.dimensiones_campo - 1
                    or self.target_position[0] < 0
                    or self.target_position[1] > self.p.dimensiones_campo - 1
                    or self.target_position[1] < 0
                ):
                    # print("\t\tOUT OF BOUNDS")
                    reward = self.p.rewards_values["out_of_bounds"]
                    done = True
                    self.moving = False
                    return reward, done

                for nb in self.neighbors(self, distance=self.p.harvest_radius):
                    if nb.identificador == "celda":
                        if nb.pertenencia != self.id:
                            # print("\t\tCELDA DE OTRA COSECHADORA")
                            reward = self.p.rewards_values["celda_otra"]
                            done = False
                            break
                        elif nb.isCosechado:
                            # print("\t\tCELDA COSECHADA")
                            reward = self.p.rewards_values["celda_cosechada"]
                            done = False
                            break
                        # Si celda no ha sido cosechada
                        elif not nb.isCosechado:
                            # print("\t\tNORMAL")
                            reward = self.p.rewards_values["normal"]
                            done = False
                            break
                        elif nb.identificador == "cosechadora":
                            # print("\t\COLISION")
                            reward = self.p.rewards_values["colision"]
                            done = True
                            break

                # Si está en el objetivo dejar de moverse
                self.moving = False
        else:
            if self.estado == "cosechando":
                rot = {
                    "NONE": -1,
                    "up": 0,
                    "down": 1,
                    "left": 2,
                    "right": 3,
                    "complete": 4,
                }

                self.previous_rotation = direction
                if self.current_direction == rot["up"] and direction == "left":
                    self.previous_rotation = rot["left"]
                    self.current_direction = rot["left"]
                elif self.current_direction == rot["up"] and direction == "right":
                    self.previous_rotation = rot["right"]
                    self.current_direction = rot["right"]
                elif self.current_direction == rot["down"] and direction == "left":
                    self.previous_rotation = rot["right"]
                    self.current_direction = rot["left"]
                elif self.current_direction == rot["down"] and direction == "right":
                    self.previous_rotation = rot["left"]
                    self.current_direction = rot["right"]
                elif self.current_direction == rot["left"] and direction == "up":
                    self.previous_rotation = rot["right"]
                    self.current_direction = rot["up"]
                elif self.current_direction == rot["left"] and direction == "down":
                    self.previous_rotation = rot["left"]
                    self.current_direction = rot["down"]
                elif self.current_direction == rot["right"] and direction == "up":
                    self.previous_rotation = rot["left"]
                    self.current_direction = rot["up"]
                elif self.current_direction == rot["right"] and direction == "down":
                    self.previous_rotation = rot["right"]
                    self.current_direction = rot["down"]
                elif self.current_direction == rot["up"] and direction == "down":
                    self.previous_rotation = rot["complete"]
                    self.current_direction = rot["down"]
                elif self.current_direction == rot["down"] and direction == "up":
                    self.previous_rotation = rot["complete"]
                    self.current_direction = rot["up"]
                else:
                    self.previous_rotation = rot["NONE"]

                x, y = self.pos[0], self.pos[1]
                if direction == "up":
                    # print("Up")
                    # self.target_position = np.array([x, y + 1])
                    self.target_position = np.ndarray(
                        (2,), buffer=np.array([x, y + 1]), dtype=int
                    )
                elif direction == "down":
                    # print("Down")
                    # self.target_position = np.array([x, y - 1])
                    self.target_position = np.ndarray(
                        (2,), buffer=np.array([x, y - 1]), dtype=int
                    )
                elif direction == "left":
                    # print("Left")
                    # self.target_position = np.array([x - 1, y])
                    self.target_position = np.ndarray(
                        (2,), buffer=np.array([x - 1, y]), dtype=int
                    )
                elif direction == "right":
                    # print("Right")
                    # self.target_position = np.array([x + 1, y])
                    self.target_position = np.ndarray(
                        (2,), buffer=np.array([x + 1, y]), dtype=int
                    )
                self.moving = True

                if (
                    self.target_position[0] > self.p.dimensiones_campo - 1
                    or self.target_position[0] < 0
                    or self.target_position[1] > self.p.dimensiones_campo - 1
                    or self.target_position[1] < 0
                ):
                    # print("\t\tOUT OF BOUNDS REWARD")
                    reward = self.p.rewards_values["out_of_bounds"]
                    done = True
                    self.moving = False
                    return reward, done

        return reward, done

    def setup_pos(self, space):
        self.space = space
        self.neighbors = space.neighbors
        self.pos = space.positions[self]
        self.pos_inicial = deepcopy(self.pos)

    def cosechar(self):
        for nb in self.neighbors(self, distance=self.p.harvest_radius):
            if nb.identificador == "celda" and not nb.isCosechado:
                self.capacidad += nb.cosechar()

                if (
                    self.capacidad + self.p.densidad > self.capacidad_max
                    and self.estado == "cosechando"
                ):
                    self.velocity = 0.0
                    self.estado = "lleno"


# Clase que representa un tractor en el campo
class Tractor(ap.Agent):
    def setup(self):
        self.velocity = 1.5
        self.identificador = "tractor"
        self.target_position_x = 0
        self.target_position_y = 0
        self.moving = False

    def move(self):
        if self.moving:
            print(
                "MOVING. Pos: ",
                self.pos,
                " Target: ",
                self.target_position_x,
                ",",
                self.target_position_y,
            )
            # Calcular el vector de dirección hacia la posición objetivo
            if abs(self.pos[0] - self.target_position_x) >= 1.5:
                direccion = (
                    np.ndarray(
                        (2,), buffer=np.array([self.target_position_x, 0]), dtype=int
                    )
                    - self.pos
                )
                distancia = np.linalg.norm(direccion)
                direccion = normalize(direccion)

                # Moverse hacia la posición objetivo
                if distancia >= self.velocity:
                    self.space.move_by(self, direccion * self.velocity)
            else:
                print("ELSE")
                direccion = (
                    np.ndarray(
                        (2,),
                        buffer=np.array(
                            [self.target_position_x, self.target_position_y]
                        ),
                        dtype=int,
                    )
                    - self.pos
                )
                distancia = np.linalg.norm(direccion)
                direccion = normalize(direccion)

                # Moverse hacia la posición objetivo
                if distancia >= self.velocity:
                    self.space.move_by(self, direccion * self.velocity)
                else:
                    # Si está cerca del objetivo, ajustar a la posición objetivo y dejar de moverse
                    self.moving = False
                    for nb in self.neighbors(self, distance=self.p.tractor_radius):
                        if (
                            nb.identificador == "cosechadora"
                            and nb.estado == "esperando"
                        ):
                            nb.capacidad = 0
                            nb.estado = "cosechando"
                            nb.velocity = 1.0
                            (
                                self.target_position_x,
                                self.target_position_y,
                            ) = self.pos_inicial
                            self.moving = True

        else:
            for nb in self.neighbors(self, distance=self.p.dimensiones_campo**2):
                if nb.identificador == "cosechadora" and nb.estado == "lleno":
                    self.target_position_x, self.target_position_y = nb.pos
                    self.moving = True
                    nb.estado = "esperando"
                    break

    def setup_pos(self, space):
        self.space = space
        self.neighbors = space.neighbors
        self.pos = space.positions[self]
        self.pos_inicial = deepcopy(self.pos)


# Clase principal que representa el modelo del campo
class FieldModel(ap.Model):
    def setup(self):
        self.space = ap.Space(self, shape=[self.p.size] * self.p.ndim)
        self.cosechadoras = ap.AgentList(
            self, self.p.cosechadora_population, Cosechadora
        )
        self.tractors = ap.AgentList(self, self.p.tractor_population, Tractor)

        # Dividir el campo en la cnaitdad de cosechadoras
        size_segment = self.p.dimensiones_campo // self.p.cosechadora_population
        positions = []
        for i in range(self.p.cosechadora_population):
            positions.append(
                np.ndarray((2,), buffer=np.array([i * size_segment, 0]), dtype=int)
            )

        self.space.add_agents(
            self.cosechadoras,
            positions=positions,
        )
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
        contador = size_segment
        cosechadora_index = 0
        for x in range(grid_size):
            if contador == 0 and cosechadora_index < self.p.cosechadora_population - 1:
                contador = size_segment
                cosechadora_index += 1
            for y in range(grid_size):
                index = x * grid_size + y
                celda = self.celdas_campo[index]
                self.space.positions[celda] = (x, y)

                # Asignar pertenencia a cada celda de campo con respecto al segmento en el que este
                celda.setup_pertenencia(self.cosechadoras[cosechadora_index].id)
                # print(f"({x},{y}) -> {cosechadora_index}")

            contador -= 1
        self.cosechadoras.cosechar()

    # Envia las posiciones de los agentes a través de WebSocket
    async def send_positions(self, websocket):
        positions_cosechadoras = {
            f"Cosechadora_{str(agent)}": [
                float(agent.pos[0]),
                float(agent.pos[1]),
                float(agent.previous_rotation),
                float(agent.capacidad),
            ]
            for agent in self.cosechadoras
        }
        positions_tractors = {
            f"Tractor_{str(agent)}": [
                float(agent.pos[0]),
                float(agent.pos[1]),
                float(-1),
            ]
            for agent in self.tractors
        }
        data = {
            **positions_cosechadoras,
            **positions_tractors,
        }
        await websocket.send(json.dumps(data))

    def egreedy_policy(self, q_values, state, epsilon=0.1):
        # Se verifica si el valor epsilon es mayor a un número aleatorio entre 0 y 1
        # Si es verdadero, se elige una acción aleatoria
        if np.random.random() < epsilon:
            # print("USO RANDOM")
            return np.random.choice(4)
        # Sino, se elige la mejor acción
        else:
            # print("USO QVALUES")
            return np.argmax(q_values[state])

    def reset(self):
        print("\n**************\nRESET\n**************\n")
        self.cosechadoras.setup()
        self.tractors.setup()

        for cosechadora in self.cosechadoras:
            moving = True
            while moving:
                direccion = cosechadora.pos_inicial - cosechadora.pos
                distancia = np.linalg.norm(direccion)
                direccion = normalize(direccion)

                # Moverse hacia la posición objetivo
                if distancia >= cosechadora.velocity:
                    # self.space.move_to(self, self.target_position)
                    cosechadora.space.move_by(
                        cosechadora, direccion * cosechadora.velocity
                    )
                else:
                    moving = False
            # self.space.move_to(cosechadora, cosechadora.pos_inicial)
        for tractor in self.tractors:
            moving = True
            while moving:
                # print(f"Pos de tractor {tractor.id}: {tractor.pos}")
                direccion = tractor.pos_inicial - tractor.pos
                distancia = np.linalg.norm(direccion)
                direccion = normalize(direccion)

                # Moverse hacia la posición objetivo
                if distancia >= tractor.velocity:
                    # self.space.move_to(self, self.target_position)
                    tractor.space.move_by(tractor, direccion * tractor.velocity)
                else:
                    moving = False
            # self.space.move_to(tractor, tractor.pos_inicial)
        for celda in self.celdas_campo:
            celda.isCosechado = False

        self.cosechadoras.cosechar()

    def q_learning(self):
        print("\n**************\nQLEARNING\n**************\n")
        num_states = self.p.dimensiones_campo**2
        num_actions = 4
        q_values = np.zeros((num_states, num_actions))
        ep_rewards = []
        direcciones = ["up", "down", "left", "right"]
        done = False
        reward_sum = 0

        for i in range(self.p.num_episodes):
            self.reset()
            done = False
            reward_sum = 0
            print(f"Episodio {i}")
            while not done:
                for cosechadora in self.cosechadoras:
                    if cosechadora.estado == "lleno":
                        continue

                    state = (
                        cosechadora.pos[1] * self.p.dimensiones_campo
                        + cosechadora.pos[0]
                    )

                    action = self.egreedy_policy(
                        q_values, state, self.p.exploration_rate_upper
                    )

                    reward = 0
                    while reward == 0 and cosechadora.velocity != 0.0:
                        # print("CICLOOOO")
                        reward, done = cosechadora.move(direcciones[action])

                    if cosechadora.estado == "lleno":
                        continue

                    next_state = (
                        cosechadora.pos[1] * self.p.dimensiones_campo
                        + cosechadora.pos[0]
                    )

                    if cosechadora.pos[0] % 2 == 0 and action == 0:
                        reward += self.p.rewards_values["up"]
                    elif cosechadora.pos[0] % 2 == 1 and action == 1:
                        reward += self.p.rewards_values["down"]

                    if (
                        cosechadora.pos[1] == self.p.dimensiones_campo - 1
                        or cosechadora.pos[1] == 0
                    ) and action == 3:
                        reward += self.p.rewards_values["sides"]

                    reward_sum += reward

                    # La ecuación de Bellman se define como:
                    # Q(s,a) = r + gamma * max(Q(s',a')) - Q(s,a)
                    if done:
                        td_target = reward
                    else:
                        td_target = reward + self.p.gamma * np.max(q_values[next_state])
                    td_error = td_target - q_values[state][action]
                    # Actualiza el valor Q para el estado y acción actuales con el valor
                    # de la ecuación de Bellman.
                    q_values[state][action] += self.p.learning_rate * td_error

                self.cosechadoras.cosechar()
                self.tractors.move()

            if self.p.exploration_rate_upper > self.p.exploration_rate_lower:
                self.p.exploration_rate_upper -= self.p.exploration_rate_decrease

            ep_rewards.append(reward_sum)

        print(f"EPISODE REWARDS: {ep_rewards}, \nQ_VALUES: \n{q_values}")

        return ep_rewards, q_values

    # Manejador de WebSocket para la simulación
    async def ws_handler_inner(self, websocket, q_values):
        print("\n**************\nSIMULACION\n**************\n")
        try:
            self.reset()
            done = False
            direcciones = ["up", "down", "left", "right"]

            while not done:
                for cosechadora in self.cosechadoras:
                    state = (
                        cosechadora.pos[1] * self.p.dimensiones_campo
                        + cosechadora.pos[0]
                    )
                    action = self.egreedy_policy(q_values, state, 0.0)

                    reward = 0
                    while reward == 0:
                        reward, done = self.cosechadoras.move(direcciones[action])[0]
                    print(direcciones[action])

                self.cosechadoras.cosechar()
                self.tractors.move()
                await self.send_positions(websocket)
                await asyncio.sleep(0.1)  # Ajustar la frecuencia según sea necesario
        except websockets.exceptions.ConnectionClosed:
            pass

    # Manejador principal de WebSocket
    async def ws_handler(self, websocket, path, q_values):
        await self.ws_handler_inner(websocket, q_values)

    # Ejecuta la simulación y WebSocket
    async def run_simulation_with_websocket(self, q_values):
        loop = asyncio.get_running_loop()

        # Habilitar tracemalloc dentro del bucle de eventos
        tracemalloc.start()

        # Inicia el servidor WebSocket
        server = await websockets.serve(
            lambda ws, path: self.ws_handler(ws, path, q_values),
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
    "size": 20,
    "dimensiones_campo": 20,
    "seed": 123,
    "steps": 1000,
    "ndim": 2,
    "densidad": 10,
    "capacidad_max": 200,
    "cosechadora_population": 3,
    "tractor_population": 5,
    "inner_radius": 1,  # 3
    "outer_radius": 3,  # 10
    "harvest_radius": 0.2,  # 1
    "border_distance": 1,  # 10
    "tractor_radius": 2,
    "cohesion_strength": 0.005,
    "separation_strength": 0.1,
    "alignment_strength": 0.3,
    "border_strength": 0.5,
    # QLEARNING
    "exploration_rate_upper": 0.1,
    "exploration_rate_lower": 0.1,
    "exploration_rate_decrease": 0.05,
    "num_episodes": 5,
    "learning_rate": 0.1,  # 0.5
    "gamma": 0.9,
    "rewards_values": {
        "normal": -1,
        "celda_cosechada": -5,
        "celda_otra": -10,
        "colision": -100,
        "out_of_bounds": -100,
        "up": 4,
        "down": 4,
        "sides": 2,
    },
}

model = FieldModel(parameters2D)
model.setup()
q_learning_rewards, q_values = model.q_learning()

model_2 = FieldModel(parameters2D)
model_2.setup()
asyncio.run(model_2.run_simulation_with_websocket(q_values))
