# Multiagentes_john_deere

## Descripción del problema
Movilidad urbana es fundamental para la vida diaria de millones de personas. Sin embargo, cuando pensamos en movilidad, a menudo pasamos por alto la importancia de la logística en la agricultura. Cada día, nuestros alimentos hacen un viaje desde el campo hasta nuestra mesa, y el proceso detrás de esto es esencial. Optimizar las operaciones agrícolas a través de un sistema inteligente no sólo mejora la eficiencia en la producción de alimentos, sino que también puede reducir costos, lo que podría traducirse en precios más bajos para los consumidores. Además, al tener una logística más eficiente, disminuimos la congestión en las carreteras, lo que beneficia a todos los que viven y trabajan en áreas urbanas.

## Institución y materia
- Instituto Tecnológico y de Estudios Superiores de Monterrey, Campus Monterrey
- Modelación de Sistemas Multiagentes con Gráficas Computacionales, Grupo 302

## Profesores asesores
- Raul V. Ramirez Velarde
- Luis Alberto Muñoz Ubando

## Equipo
- Ainhize Legarreta García
- Enrique Uzquiano Puras
- Christopher Pedraza Pohlenz
- David Cavazos Wolberg
- Fausto Pacheco Martínez

## Presentación
- [Link a la presentación en Canva](https://www.canva.com/design/DAF0D-UutKo/DtBF1n_CguwKEUeysDT6Mw/view?utm_content=DAF0D-UutKo&utm_campaign=designshare&utm_medium=link&utm_source=editor)

## Demos
- [Conexión Unity-Python con websocket e interacción multiagente](https://youtu.be/0q4Wyd9vReg)
- [Implementación de Q-Learning para el entrenamiento](https://youtu.be/KeXsPdduUNk)
- [Simulación final](https://youtu.be/zFzxAbX7jCE)

## Librerías necesarias para correr el programa de Python
- pymemtrace
```
pip install pymemtrace
```
- asyncio
```
pip install asyncio
```
- websockets
```
pip install websockets
```
- agentpy
```
pip install agentpy
```
- numpy
```
pip install numpy
```

## Librería usada en simulación de Unity
- [websocket-sharp](https://github.com/sta/websocket-sharp)

## ¿Cómo correr la simulación?
1. Primero es necesario entrenar el sistema. Esto se puede hacer corriendo cualquiera de los dos archivos en la carpeta de [Simulacion Multiagente](https://github.com/christopher-pedraza/multiagentes_john_deere/tree/0e7eb4713ab3565cda7ad0c8203ce414ded7bba3/Sistema%20Multiagente). La diferencia es que el script "SistemaMultiagente_QLearning_Normal_Training.py" entrena el modelo sin hacer ningúna conexión con el websocket y solo se conecta hasta que empieza la simulación ya entrenada. Por otro lado, "SistemaMultiagente_QLearning_Unity_Training.py" despliega en Unity los movimientos de los agentes mientras se entrena. Esta segunda opción es mucho más lenta y tiene el problema que el websocket se desconecta antes de que se acaben los episodios, por lo que no termina de entrenar el sistema. Sin embargo, permite apreciar cómo están interactuando los agentes al inicio mientras se entrenan. Es recomendado usar la primer opción para poder entrenar completamente el modelo y que sea más rápido.
2. Cuando el script de Python despliegue la tabla Q, se quedará esperando a que se conecte un cliente para empezar a mandar a correr la simulación y mandar las coordenadas de los agentes por el socket.
3. Corre la simulación desde Unity. Es importante que se esté en la escena "Main", por lo que si al correr la simulación no aparece el campo, puede que no se encuentre en la escena correcta. Para cambiar de escena es necesario entrar a la carpeta `Assets/Scenes/` desde la ventana de Project en el editor de Unity y dar doble click a la escena "Main".
4. Disfruta de ver como se cosecha el campo!! Recuerda que se pueden modificar los parámetros desde el script de Python si deseas experimentar más.
