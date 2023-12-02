using UnityEngine;

public class wheelTurn : MonoBehaviour
{
    public float spinSpeed = 13f; // Adjust the spin speed as needed

    void Update()
    {
        // Rotate the object continuously around its up axis
        transform.Rotate(Vector3.right, spinSpeed * Time.deltaTime);
    }
}
