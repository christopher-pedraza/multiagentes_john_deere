using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlaneGridGenerator : MonoBehaviour {
    public GameObject planePrefab; // Prefab for the plane
    public int gridWidth = 5;      // Number of planes in the X direction
    public int gridHeight = 5;     // Number of planes in the Z direction
    public float spacing = 1.0f;   // Spacing between planes

    void Start() {
        GeneratePlaneGrid();
    }

    void GeneratePlaneGrid() {
        for (int i = 0; i < gridWidth; i++) {
            for (int j = 0; j < gridHeight; j++) {
                // Calculate the position for each plane based on the grid and spacing
                Vector3 position = new Vector3(i * spacing, 0, j * spacing) + transform.position;

                // Instantiate a plane prefab at the calculated position
                GameObject plane = Instantiate(planePrefab, position, Quaternion.identity);

                // Set the parent of the instantiated plane to the empty GameObject
                plane.transform.parent = transform;
            }
        }
    }
}

