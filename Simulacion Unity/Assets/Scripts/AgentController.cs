using System;
using System.Collections;
using UnityEngine;
using WebSocketSharp;
using Newtonsoft.Json;
using System.Collections.Generic;

public class AgentController : MonoBehaviour
{
    // Prefabs para agentes en Unity
    public GameObject cosechadoraPrefab;
    public GameObject tractorPrefab;

    // Conexión WebSocket y diccionario para almacenar agentes
    private WebSocket ws;
    private Dictionary<string, GameObject> agents = new Dictionary<string, GameObject>();

    // Método Start que se ejecuta al inicio
    void Start()
    {
        Debug.Log("Conectando a WebSocket...");
        ws = new WebSocket("ws://localhost:8765");
        ws.OnMessage += OnMessage;
        ws.OnError += OnError;
        ws.OnClose += OnClose;

        agents = new Dictionary<string, GameObject>(); // Inicializa el diccionario

        // Intenta conectar
        ws.Connect();

        // Verifica si la conexión fue exitosa
        if (ws.ReadyState == WebSocketState.Open)
        {
            // Conexión exitosa, inicia la rutina de actualización
            StartCoroutine(UpdateAgents());
        }
        else
        {
            // Fallo de conexión, registra un error
            Debug.LogError("La conexión WebSocket falló.");
        }
    }

    // Maneja los mensajes recibidos a través de WebSocket
    void OnMessage(object sender, MessageEventArgs e)
    {
        try
        {
            // Registra los datos JSON recibidos
            Debug.Log("Datos JSON recibidos: " + e.Data);

            // Parsea los datos JSON usando JSON.NET
            Dictionary<string, List<float>> agentsData;
            try
            {
                agentsData = JsonConvert.DeserializeObject<Dictionary<string, List<float>>>(e.Data);
            }
            catch (Exception ex)
            {
                Debug.LogError("Error al analizar los datos JSON: " + ex);
                return;
            }

            // Registra el número de agentes recibidos
            Debug.Log($"Número de agentes recibidos: {agentsData.Count}");

            // Recorre los agentes y actualiza las posiciones
            foreach (var entry in agentsData)
            {
                string agentName = entry.Key;
                List<float> positionData = entry.Value;

                // Extrae la información del tipo (asumiendo que 'Cosechadora' o 'Tractor' está incluido en agentName)
                string agentType = "";
                if (agentName.StartsWith("Cosechadora"))
                {
                    agentType = "Cosechadora";
                }
                else if (agentName.StartsWith("Tractor"))
                {
                    agentType = "Tractor";
                }
                else if (agentName.StartsWith("Campo"))
                {
                    agentType = "Campo";
                }
                else if (agentName.StartsWith("Rotacion"))
                {
                    agentType = "Rotacion";
                }


                if (!agentType.Equals("Campo"))
                {
                    // Registra información para depuración
                    Debug.Log($"Agente: {agentName}, Tipo: {agentType}, Posición: {positionData[0]}, {positionData[1]}");

                    MainThreadDispatcher.Instance.DispatchToMainThread(() =>
                    {
                        // Verifica si el agente ya está en el diccionario
                        if (agents.ContainsKey(agentName))
                        {
                            // Actualiza la posición del agente existente
                            agents[agentName].transform.position = new Vector3(positionData[0], 0.0f, positionData[1]);

                            Debug.Log($"Actualizando posición del agente existente: {agentName}");
                            // Izquierda
                            if (positionData[2] == 2)
                            {
                                agents[agentName].transform.Rotate(0, 90, 0);
                                Debug.Log($"{agentName} roto a la: IZQUIERDA");
                            }
                            // Derecha
                            else if (positionData[2] == 3)
                            {
                                agents[agentName].transform.Rotate(0, -90, 0);
                                Debug.Log($"{agentName} roto a la: DERECHA");
                            }
                            // 180
                            else if (positionData[2] == 4)
                            {
                                agents[agentName].transform.Rotate(0, 90, 0);
                                agents[agentName].transform.Rotate(0, 90, 0);
                                Debug.Log($"{agentName} roto a la: 180");
                            }
                        }
                        else
                        {
                            if (cosechadoraPrefab != null && tractorPrefab != null)
                            {
                                // Instancia un nuevo agente según el tipo
                                GameObject newAgent;
                                if (agentType == "Cosechadora")
                                {
                                    newAgent = Instantiate(cosechadoraPrefab, new Vector3(positionData[0], 0.0f, positionData[1]), Quaternion.identity);
                                }
                                else if (agentType == "Tractor")
                                {
                                    newAgent = Instantiate(tractorPrefab, new Vector3(positionData[0], 0.0f, positionData[1]), Quaternion.identity);
                                }
                                else
                                {
                                    Debug.LogError($"Tipo de agente desconocido: {agentType}");
                                    return;
                                }

                                agents.Add(agentName, newAgent);
                                Debug.Log($"Instanciando nuevo agente: {agentName}");
                            }
                        }
                    });
                }
            }
        }
        catch (Exception ex)
        {
            Debug.LogError("Excepción en OnMessage: " + ex);
        }
    }

    // Maneja errores en la conexión WebSocket
    void OnError(object sender, ErrorEventArgs e)
    {
        Debug.LogError("Error en WebSocket: " + e.Message);
    }

    // Maneja el cierre de la conexión WebSocket
    void OnClose(object sender, CloseEventArgs e)
    {
        Debug.Log("WebSocket cerrado. Código: " + e.Code + ", Razón: " + e.Reason);
    }

    // Rutina que actualiza periódicamente los agentes
    IEnumerator UpdateAgents()
    {
        while (true)
        {
            Debug.Log("Actualizando agentes...");

            // Verifica si la conexión WebSocket está abierta antes de enviar datos
            if (ws != null && ws.ReadyState == WebSocketState.Open)
            {
                // Solicita las posiciones de los agentes al servidor
                ws.Send("get_positions");
            }
            else
            {
                Debug.LogWarning("La conexión WebSocket no está abierta.");
            }

            yield return new WaitForSeconds(0.5f); // Ajusta la frecuencia de actualizaciones según sea necesario
        }
    }

    // Método que se llama cuando se destruye el objeto
    private void OnDestroy()
    {
        if (ws != null)
        {
            ws.Close();
        }
    }
}

// Clase para representar datos de un agente
[System.Serializable]
public class AgentData
{
    public List<float> position;

    // Agrega una propiedad para obtener Vector2 a partir de la lista
    public Vector2 GetVector2Position()
    {
        if (position != null && position.Count >= 2)
        {
            return new Vector2(position[0], position[1]);
        }
        return Vector2.zero; // Devuelve un valor predeterminado si la lista no es válida
    }
}

// Clase para representar datos de varios agentes
[Serializable]
public class AgentsData
{
    public Dictionary<string, List<float>> agents;
}
