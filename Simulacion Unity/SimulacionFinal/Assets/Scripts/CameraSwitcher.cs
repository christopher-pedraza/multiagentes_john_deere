using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using UnityEngine;

public class CameraSwitcher : MonoBehaviour
{
    public Camera camera1;
    public Camera camera2;
    public Camera camera3;

    private int activeCameraIndex = 0;

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.L))
        {
            activeCameraIndex = (activeCameraIndex + 1) % 3;
            SwitchCamera(activeCameraIndex);
        }
    }

    private void SwitchCamera(int index)
    {
        camera1.enabled = (index == 0);
        camera2.enabled = (index == 1);
        camera3.enabled = (index == 2);
    }
}

