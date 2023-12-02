using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MeshGeneration : MonoBehaviour
{

    Mesh mesh;

    Vector3[] vertices;
    int[] triangles;
    // Start is called before the first frame update
    void Start()
    {
        mesh = new Mesh();
        GetComponent<MeshFilter>().mesh = mesh;

        CreateShape();
        UpdateMesh();
    }

    void CreateShape(){

    vertices = new Vector3[]{
        new Vector3(0, 0, 0),
            new Vector3(1, 0, 0),
            new Vector3(0, 0, 1),
            new Vector3(1, 0, 1),

            new Vector3(0, 1, 0),
            new Vector3(1, 1, 0),
            new Vector3(0, 1, 1),
            new Vector3(1, 1, 1),
    };

    triangles = new int[]{
         0, 2, 1,
            2, 3, 1,

            4, 5, 6,
            6, 5, 7,

            0, 1, 4,
            1, 5, 4,

            2, 6, 3,
            3, 6, 7,

            0, 4, 2,
            2, 4, 6,

            1, 3, 5,
            3, 7, 5,
        };
    }

    // Update is called once per frame
    void UpdateMesh()
    {
        mesh.Clear();
        mesh.vertices=vertices;
        mesh.triangles=triangles;
        mesh.RecalculateNormals();

    }
}
