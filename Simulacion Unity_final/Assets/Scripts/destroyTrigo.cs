using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class destroyTrigo : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        
    }

    private void OnTriggerEnter(Collider other)
            {
            if (other.CompareTag("Planta") )
                Destroy(other.gameObject);
            }

    // Update is called once per frame
    void Update()
    {
        
    }
}
