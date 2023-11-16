# Multiagentes_john_deere

![Tec de Monterrey](TecLogo.png)


**Instituto Tecnológico y de Estudios Superiores de Monterrey**

**Campus Monterrey**

**TC2008B - Grupo 302**

**Modelación de Sistemas Multiagentes con Gráficas Computacionales**


**Profesores:**
- Raul V. Ramirez Velarde
- Luis Alberto Muñoz Ubando

**Equipo 3:**
- Ainhize Legarreta García (A01762291)
- Enrique Uzquiano Puras (A01762083)
- Christopher Pedraza Pohlenz (A01177767)
- David Cavazos Wolberg (A01721909)
- Fausto Pacheco Martínez (A01412004)

## Índice
1. [Interpretación del Reto](#interpretación-del-reto)
2. [Parámetros Considerados](#parámetros-considerados)
   - [Determinados por el Usuario](#determinados-por-el-usuario)
   - [Valores Aleatorios](#valores-aleatorios)
   - [Valores Estáticos](#valores-estáticos)
3. [Objetivos](#objetivos)
4. [Algoritmos](#algoritmos)
   - [Algoritmos Ávaros/Greedy](#algoritmos-ávaros-o-greedy)
   - [Algoritmos de Búsqueda de Caminos Más Cortos](#algoritmos-de-búsqueda-de-caminos-más-cortos)
5. [Lógica](#lógica)
   - [Cosechadora](#cosechadora)
   - [Tractor](#tractor)
   - [Miró](#miró)
6. [Simulación](#simulación)
   - [Link al Video de Simulación](#link-al-video-de-simulación)
7. [Presentación](#presentación)
   - [Link al Script de Presentación en Canva](#link-al-script-de-presentación-de-canva)
8. [Camino a Seguir/Plan](#camino-a-seguir-plan)

## Interpretación del Reto
Se tendrá un campo y una serie de tractores y cosechadoras. Todo esto contará con parámetros que podrán ser especificados por el usuario previo a la simulación.

## Parámetros Considerados
### Determinados por el Usuario:
- Longitud y ancho del campo
- Cantidad de cosechadoras y tractores
- Velocidad base de la cosechadora y tractor
- Capacidad máxima de la cosechadora y tractor

### Valores Aleatorios:
- Posición de inicio de los tractores y cosechadoras
- Pendientes dentro del terreno irregular (se calcularán aleatoriamente dentro de un rango)

### Valores Estáticos:
- Radio de cosecha de cada cosechadora

## Objetivos
- Minimizar el tiempo de cosecha mientras se maximiza el campo cosechado
- Optimizar las rutas de cosecha
- Llevar un seguimiento de la gasolina usada, velocidad de los vehículos y cantidad cosechada por metro cuadrado en tiempo real.

## Algoritmos
Para poder solucionar el problema, se plantearon diversos algoritmos que se podrían implementar para poder determinar las rutas más óptimas de cada agente (tractores y cosechadoras).

### Algoritmos Ávaros/Greedy
En este enfoque, el algoritmo toma decisiones en cada paso basándose únicamente en la información seleccionando la opción que parece la mejor en ese instante sin considerar las consecuencias a largo plazo.

### Algoritmos de Búsqueda de Caminos Más Cortos (A*, BFS, Dijkstra)
En estos algoritmos, se explora un espacio de búsqueda que generalmente se representa como un grafo, donde los nodos son estados o posiciones y los vínculos son transiciones entre estos estados.

- Búsqueda en Amplitud (BFS): Explora el grafo en capas, desde el nodo inicial hacia fuera, asegurándose de que se exploran todas las posibilidades a la misma profundidad antes de profundizar más.
- Dijkstra: Encuentra el camino más corto en un grafo con pesos no negativos, explorando primero las rutas más cortas.
- A*: Combina elementos de BFS y Dijkstra, utilizando una función heurística para priorizar nodos que parecen prometedores, lo que lo hace eficiente en la búsqueda de rutas cortas en grafos con pesos.

Para comparar y verificar la calidad de las soluciones obtenidas, es una buena estrategia comenzar con algoritmos más simples como BFS, que garantiza encontrar la solución óptima en grafos con pesos no negativos.

## Lógica
### Cosechadora
(Descripción de la lógica de la cosechadora)

### Tractor
(Descripción de la lógica del tractor)

### Miró
[Miró Board](https://miro.com/app/board/uXjVNR0RZDw=/?share_link_id=372821292186)

## Simulación
[Link al Video de Simulación](#)

## Presentación
[Link al Script de Presentación en Canva](https://www.canva.com/design/DAF0D-UutKo/DtBF1n_CguwKEUeysDT6Mw/view?utm_content=DAF0D-UutKo&utm_campaign=designshare&utm_medium=link&utm_source=editor)

## Camino a Seguir/Plan
1. Definir el problema
2. Lógica
3. Código
   - Crear ambiente donde se hará la simulación
   - Crear agentes
   - Decisiones para cada agente
   - Obtener puntuaciones en cada posición que representarán qué tan cerca está del objetivo (goal).
4. Gráfica Computacional
