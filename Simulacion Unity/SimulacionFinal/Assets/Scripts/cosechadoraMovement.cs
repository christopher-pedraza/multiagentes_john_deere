using System.Collections;
using System.Collections.Generic;
using UnityEngine;



public class cosechadoraMovement : MonoBehaviour
{
    public float moveSpeed;
    public float rotationSpeed; // Slower rotation speed
    private Quaternion targetRotation;
    public float smoothTime = 5.0f;
    public float movementDuration = 5.0f;

    private Vector3 startPosition;
    private Vector3 endPosition;
    private float movementStartTime;

    private bool isMoving;


    IEnumerator MoveForwardForDistance(float distance)
    {
        isMoving = true;
        while (transform.position.z < distance)
        {
            transform.Translate(Vector3.forward * moveSpeed * Time.deltaTime);
            yield return null;
        }

        isMoving = false;
    }


    void MoveForward()
    {
       // Vector3 movement = transform.forward * 5.0f;
       // transform.Translate(movement * 5.0f * movementDuration, Space.World);
        

        
        StartCoroutine(MoveForwardForDistance(15.0f));
        //transform.Translate(Vector3.forward *  moveSpeed * Time.deltaTime);


    }

    void MoveBackward()
    {
        startPosition = transform.position;
        endPosition = transform.position - transform.forward * 5.0f;
    }

    void TurnRight()
    {
        targetRotation *= Quaternion.Euler(0, 90, 0);
        MoveForward(); // Optional: combine turn with forward movement
    }

    void TurnLeft()
    {
        targetRotation *= Quaternion.Euler(0, -90, 0);
        MoveForward(); // Optional: combine turn with forward movement
    }

    void Start()
    {
        targetRotation = transform.rotation; // Initialize with current rotation
    }

    void Update()
    {

       
       

        float verticalInput = Input.GetAxis("Vertical");
        
        // Handle rotation
        if (Input.GetKeyDown(KeyCode.RightArrow) || Input.GetKeyDown(KeyCode.D))
        {
            TurnRight();
        }
        else if (Input.GetKeyDown(KeyCode.LeftArrow) || Input.GetKeyDown(KeyCode.A))
        {
            TurnLeft();
        }
        else if (Input.GetKeyDown(KeyCode.UpArrow) || Input.GetKeyDown(KeyCode.W))
        {
            Debug.Log("UP");
            MoveForward();
        }
        else if (Input.GetKeyDown(KeyCode.DownArrow) || Input.GetKeyDown(KeyCode.S))
        {
            MoveBackward();
        }

    }

    
    

}




