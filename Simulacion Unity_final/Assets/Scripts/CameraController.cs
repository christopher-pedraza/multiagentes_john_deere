using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using UnityEngine;

public class CameraController : MonoBehaviour
{
    public GameObject field; // Reference to the field GameObject
    public float angle = 25f; // Desired angle in degrees

    void Update()
    {
        AdjustCamera();
    }

    void AdjustCamera()
    {
        // Calculate the size of the field
        float fieldSize = 785; // Determine the field size based on your field generation logic

        // Calculate the required height based on field size and desired angle
        float height = CalculateCameraHeight(fieldSize, angle);

        // Set the camera position
        transform.position = new Vector3(fieldSize / 2, height, fieldSize / 2);

        // Set the camera rotation
        transform.rotation = Quaternion.Euler(angle, 0, 0);
    }

    float CalculateCameraHeight(float fieldSize, float angle)
    {
        // Convert angle to radians
        float angleInRadians = 24;

        // Calculate the height
        float height = fieldSize * Mathf.Tan(angleInRadians);

        return height;
    }
}
