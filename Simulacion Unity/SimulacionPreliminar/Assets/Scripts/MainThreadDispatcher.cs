using System;
using UnityEngine;
using System.Collections.Generic;

public class MainThreadDispatcher : MonoBehaviour
{
    private static MainThreadDispatcher _instance;

    public static MainThreadDispatcher Instance
    {
        get
        {
            if (_instance == null)
            {
                GameObject go = new GameObject("MainThreadDispatcher");
                _instance = go.AddComponent<MainThreadDispatcher>();
            }
            return _instance;
        }
    }

    private readonly Queue<Action> _actions = new Queue<Action>();
    private object _lock = new object();

    private void Awake()
    {
        if (_instance != null && _instance != this)
        {
            Destroy(gameObject);
            return;
        }

        _instance = this;
        DontDestroyOnLoad(gameObject);
    }

    private void Update()
    {
        lock (_lock)
        {
            while (_actions.Count > 0)
            {
                _actions.Dequeue()?.Invoke();
            }
        }
        Debug.Log("MainThreadDispatcher Update");
    }

    public void DispatchToMainThread(Action action)
    {
        lock (_lock)
        {
            _actions.Enqueue(action);
        }
    }
}
