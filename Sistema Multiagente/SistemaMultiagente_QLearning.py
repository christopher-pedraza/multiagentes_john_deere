import tracemalloc
import asyncio
import websockets
import agentpy as ap
import numpy as np
import json
import pickle

import os

# Add this at the beginning of your script
os.chdir("D:\Folders\Desktop\multiagentes_john_deere\Sistema Multiagente")

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
    def setup(self, unique_id):
        print(f"Cosechadora setup, ID: {unique_id}")

        self.velocity = 1
        self.identificador = "cosechadora"
        self.capacidad_max = self.p.capacidad_max
        self.capacidad = 0
        self.estado = "cosechando"
        self.unique_id = unique_id

        # Set the initial position for cosechadoras along the bottom side
        self.pos = np.array([0, unique_id * 10])  # Adjust the spacing as needed

        # Call setup_pos to ensure the position is set in the space
        self.setup_pos(self.model.space)

        # Q-learning parameters
        self.learning_rate = self.p.learning_rate
        self.discount_factor = self.p.discount_factor
        self.exploration_prob = self.p.exploration_prob
        self.q_table = {}  # Q-table to store state-action values
        self.prev_state = None
        self.prev_action = None

    def setup_pos(self, space):
        # If the position is not set during setup, use the space's positions
        if self.pos is None:
            super().setup_pos(space)
        else:
            space.positions[self] = tuple(self.pos)
            self.space = space
            self.neighbors = space.neighbors

        print(f"Cosechadora position set: {self.pos}")

    def setup_reset(self):
        # Additional setup logic for reset, if needed
        pass

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

            print("Cosechadora updating velocity")

    def update_position(self):
        if self.estado == "cosechando":
            print(f"Current Velocity: {self.velocity}")
            # self.space.move_by(self, self.velocity)
            new_position = tuple(
                np.array(self.space.positions[self]) + np.array(self.velocity)
            )
            self.space.positions[self] = new_position

            print(f"Cosechadora updating position: {self.pos}")

    def cosechar(self):
        for nb in self.neighbors(self, distance=self.p.harvest_radius):
            if nb.identificador == "celda" and not nb.isCosechado:
                self.capacidad += nb.cosechar()

                if self.capacidad + self.p.densidad > self.capacidad_max:
                    self.velocity = 0
                    self.estado = "lleno"

        # Q-learning update
        state = tuple(self.pos)
        if self.prev_state is not None and self.prev_action is not None:
            reward = self.calculate_reward(state)
            self.update_q_table(self.prev_state, self.prev_action, reward, state)

        # Choose the next action using epsilon-greedy policy
        if np.random.rand() < self.exploration_prob:
            # Randomly select one of the velocity adjustment actions
            action = np.random.choice(
                ["move_left", "move_right", "move_up", "move_down"]
            )
        else:
            # Choose the action with the highest Q-value
            action = max(
                self.q_table.get(
                    state,
                    {"move_left": 0, "move_right": 0, "move_up": 0, "move_down": 0},
                ),
                key=self.q_table.get(
                    state,
                    {"move_left": 0, "move_right": 0, "move_up": 0, "move_down": 0},
                ).get,
            )

        self.prev_state = state
        self.prev_action = action

        print("Cosechadora cosechando")

        return self.perform_action(action)

    def calculate_reward(self, state):
        print("Calculating reward")

        # Calculate the reward based on the state
        # For example, penalize revisiting a harvested cell
        if self.prev_state in self.q_table and state == self.prev_state:
            return -10  # Penalize revisiting
        return 0  # Default reward

    def update_q_table(self, state, action, reward, next_state):
        print(f"Updating Q-table for Cosechadora {self.unique_id}")
        print(
            f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}"
        )

        # Ensure self.q_table is initialized
        if self.q_table is None:
            self.q_table = {}

        # Update Q-table based on the Q-learning update rule
        q_value = self.q_table.get((state, action), 0)
        next_max_q_value = max(self.q_table.get(next_state, {}).values(), default=0)
        new_q_value = q_value + self.learning_rate * (
            reward + self.discount_factor * next_max_q_value - q_value
        )

        # Update the Q-table entry for the given state and action
        self.q_table[(state, action)] = new_q_value

        # Ensure there is an entry for the next_state in the Q-table
        if next_state not in self.q_table:
            self.q_table[next_state] = {
                "move_left": 0,
                "move_right": 0,
                "move_up": 0,
                "move_down": 0,
            }

        print(f"Updated Q-table: {self.q_table}")

    def perform_action(self, action):
        print(f"Cosechadora performing action: {action}")

        # Perform the selected action by adjusting the velocity
        velocity_change = self.p.velocity_change
        if action == "move_left":
            self.velocity = np.array([-velocity_change, 0])
        elif action == "move_right":
            self.velocity = np.array([velocity_change, 0])
        elif action == "move_up":
            self.velocity = np.array([0, -velocity_change])
        elif action == "move_down":
            self.velocity = np.array([0, velocity_change])
        return 0

    def save_q_values(self, filename):
        print(f"Saving Q-values for Cosechadora {self.unique_id}")
        with open(filename, "wb") as file:
            pickle.dump(self.q_table, file)

    def load_q_values(self, filename):
        print("Loading Q-values")
        with open(filename, "rb") as file:
            self.q_table = pickle.load(file)


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
    def __init__(self, parameters):
        super().__init__(parameters)
        self.space = ap.Space(self, shape=[self.p.size] * self.p.ndim)

        # Create celdas_campo without adding them to the space yet
        self.celdas_campo = ap.AgentList(
            self, self.p.dimensiones_campo**2, CeldaCampo
        )

        # Create cosechadoras and add them to the AgentList
        self.cosechadoras = ap.AgentList(
            self,
            [
                Cosechadora(self, unique_id=i)
                for i in range(self.p.cosechadora_population)
            ],
        )

        # Print the number of cosechadoras to verify
        print(f"Number of cosechadoras: {len(self.cosechadoras)}")

        self.tractors = ap.AgentList(self, self.p.tractor_population, Tractor)
        self.space.add_agents(self.cosechadoras, random=True)
        self.space.add_agents(self.tractors, random=True)
        self.cosechadoras.setup_pos(self.space)
        self.tractors.setup_pos(self.space)

        # Assign manually positions to each celda_campo
        grid_size = self.p.dimensiones_campo
        for x in range(grid_size):
            for y in range(grid_size):
                index = x * grid_size + y
                celda = self.celdas_campo[index]
                self.space.positions[celda] = (x, y)

        self.space.add_agents(self.celdas_campo)

    # def setup(self):
    #     print("Model setup")

    #     self.space = ap.Space(self, shape=[self.p.size] * self.p.ndim)
    #     self.cosechadoras = ap.AgentList(
    #         [
    #             Cosechadora(self, unique_id=i)
    #             for i in range(self.p.cosechadora_population)
    #         ]
    #     )
    #     # Manually assign unique IDs during agent creation
    #     for unique_id in range(self.p.cosechadora_population):
    #         self.cosechadoras.add(unique_id=unique_id)

    #     self.tractors = ap.AgentList(self, self.p.tractor_population, Tractor)
    #     self.space.add_agents(self.cosechadoras, random=True)
    #     self.space.add_agents(self.tractors, random=True)
    #     self.cosechadoras.setup_pos(self.space)
    #     self.tractors.setup_pos(self.space)

    #     # Crear celdas_campo sin agregarlas al espacio aún
    #     self.celdas_campo = ap.AgentList(
    #         self, self.p.dimensiones_campo**2, CeldaCampo
    #     )

    #     # Asignar manualmente posiciones a cada celda_campo
    #     grid_size = self.p.dimensiones_campo
    #     for x in range(grid_size):
    #         for y in range(grid_size):
    #             index = x * grid_size + y
    #             celda = self.celdas_campo[index]
    #             self.space.positions[celda] = (x, y)

    #     self.space.add_agents(self.celdas_campo)

    async def run_qlearning_episode(self, episode):
        print(f"Running Q-learning episode {episode}")

        # Reset the environment for a new episode
        self.reset_environment()

        for step in range(self.p.qlearning_steps):
            self.step()
            await asyncio.sleep(0.1)

            # Additional logic for Q-learning exploration
            for cosechadora in self.cosechadoras:
                cosechadora.learning_rate = self.p.learning_rate
                cosechadora.discount_factor = self.p.discount_factor
                cosechadora.exploration_prob = self.p.exploration_prob

                # Record the current state, action, and reward
                state = tuple(cosechadora.pos)
                action = cosechadora.prev_action
                reward = cosechadora.calculate_reward(state)

                # Update Q-values based on the Q-learning update rule
                cosechadora.update_q_table(
                    cosechadora.prev_state, action, reward, state
                )

        print("3")

        # Save Q-values after each Q-learning episode
        for cosechadora in self.cosechadoras:
            print("4")
            file_name = f"q_values_episode{episode}.pkl"
            cosechadora.save_q_values(file_name)
            print(f"Saved Q-values for episode {episode} to {file_name}")

        # Exploration decay after each episode
        self.p.exploration_prob *= self.p.exploration_decay_factor

    def reset_environment(self):
        print("Resetting environment")

        # Reset positions, velocities, and any other relevant states
        for cosechadora in self.cosechadoras:
            cosechadora.setup_reset()

        for celda in self.celdas_campo:
            celda.isCosechado = False

    # Envia las posiciones de los agentes a través de WebSocket
    async def send_positions(self, websocket):
        print("Sending positions")

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

    async def send_positions_with_q_values(self, websocket):
        try:
            while True:
                # Update positions based on Q-values
                for cosechadora in self.cosechadoras:
                    state = tuple(cosechadora.pos)
                    action = max(
                        cosechadora.q_table.get(
                            state,
                            {
                                "move_left": 0,
                                "move_right": 0,
                                "move_up": 0,
                                "move_down": 0,
                            },
                        ),
                        key=cosechadora.q_table.get(
                            state,
                            {
                                "move_left": 0,
                                "move_right": 0,
                                "move_up": 0,
                                "move_down": 0,
                            },
                        ).get,
                    )
                    cosechadora.perform_action(action)
                    cosechadora.update_position()

                # Send positions through WebSocket
                await self.send_positions(websocket)

                await asyncio.sleep(0.1)
                print("Sending positions with Q-values")
        except websockets.exceptions.ConnectionClosed:
            pass

    async def run_simulation_with_q_values(self):
        print("Running simulation with Q-values")

        # Load Q-values obtained during Q-learning episodes
        for cosechadora in self.cosechadoras:
            cosechadora.load_q_values(
                f"q_values_episode{self.p.qlearning_episodes-1}.pkl"
            )

        loop = asyncio.get_running_loop()

        # Inicia el servidor WebSocket
        server = await websockets.serve(
            lambda ws, path: self.send_positions_with_q_values(ws),
            "localhost",
            8766,  # Change the port number if needed
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
    # Q-learning parameters
    "learning_rate": 0.1,
    "discount_factor": 0.9,
    "exploration_prob": 0.1,
    "qlearning_episodes": 5,  # Number of episodes for Q-learning exploration
    "qlearning_steps": 100,  # Number of steps per episode
    "exploration_decay_factor": 0.95,  # Decay factor for exploration probability
    "velocity_change": 1,  # Magnitude of velocity change per Q-learning action
}


async def run_model():
    print("Running the model")

    # Create a model instance
    model = FieldModel(parameters2D)

    # Run multiple Q-learning episodes
    # for episode in range(model.p.qlearning_episodes):
    #     await model.run_qlearning_episode(episode)

    # Run the simulation after completing all Q-learning episodes
    task_simulation = model.run_simulation_with_q_values()
    await task_simulation


# Run the model
asyncio.run(run_model())
