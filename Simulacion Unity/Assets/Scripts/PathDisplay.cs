using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PathDisplay : MonoBehaviour {
    public float targetTime;
    float time;
    public GameObject prefab;

    // Start is called before the first frame update
    void Start() {
        
    }

    // Update is called once per frame
    void FixedUpdate() {
        time -= Time.deltaTime;

        if (time <= 0.0f) {
            Vector3 position = transform.position;
            Instantiate(prefab, position, Quaternion.identity);
            time = targetTime;
        }
    }
}
