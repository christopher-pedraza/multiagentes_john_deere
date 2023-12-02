using UnityEngine;

public class podadoraSpin : MonoBehaviour
{
    public float spinSpeed = 5; // Adjust the spin speed as needed
    public Transform object1;
    public Transform object2;

    void Update()
    {
        // Rotate the object continuously around its up axis
         // Calculate the middle point between object1 and object2
        Vector3 middlePoint = (object1.position + object2.position) / 2f;

        // Rotate the object around the middle point in the Z-axis
        transform.RotateAround(middlePoint, Vector3.right, spinSpeed * Time.deltaTime);
    }
}
