using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ColorChange : MonoBehaviour
{
    private Material originalMaterial;
    public Material highlightMaterial; // Assign this in the inspector

    private void Start() {
        originalMaterial = GetComponent<Renderer>().material;
    }

    private void OnTriggerEnter(Collider other) {
        Debug.Log(other.tag);
        if (other.CompareTag("cosechadora")) {
            // Change color to highlightMaterial
            GetComponent<Renderer>().material = highlightMaterial;
        }
    }

    // private void OnTriggerExit(Collider other) {
    //     if (other.CompareTag(tag)) {
    //         // Change color back to the originalMaterial
    //         GetComponent<Renderer>().material = originalMaterial;
    //     }
    // }
}
