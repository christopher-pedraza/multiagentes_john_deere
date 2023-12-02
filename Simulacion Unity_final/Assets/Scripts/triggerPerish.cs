using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class triggerPerish : MonoBehaviour
{
     private void OnTriggerEnter(Collider other) {
         if (other.CompareTag("algo"))
        {
            // Code to handle being harvested by the harvester
            // e.g., Deactivate itself, play an animation, etc.
            gameObject.SetActive(false);
        }


    }
}

