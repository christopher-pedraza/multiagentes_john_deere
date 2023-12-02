    using System.Collections;
    using System.Collections.Generic;
    using UnityEngine;

    public class Field : MonoBehaviour
    {
        public GameObject groundPrefab; // Ground cell prefab
        public GameObject outerLayerTop; // First outer layer cell prefab
        public GameObject outerLayerBottom; // Second outer layer cell prefab
        public GameObject outerLayerRight; // First outer layer cell prefab
        public GameObject outerLayerLeft; // Second outer layer cell prefab


        public GameObject cornerTopRight; // First outer layer cell prefab
        public GameObject cornerTopLeft; // Second outer layer cell prefab
        public GameObject cornerBottomRight; // First outer layer cell prefab
        public GameObject cornerBottomLeft; // Second outer layer cell prefab

        public int fieldSize; // Size of the square field

        void Start()
        {
            GenerateField();
        }

        void GenerateField()
        {

            int totalSize = fieldSize + 4;
            

            for (int x = 0; x < totalSize; x++)
            {
                for (int z = 0; z < totalSize; z++)
                {
                
                    
                    Vector3 position = new Vector3(x *4.315f, 0f, z*4.315f);

                    
                        GameObject prefabToUse;
                        GameObject cell;
                        bool myBool = false;

                            if(x==0 && z == 0){
                                prefabToUse = cornerTopLeft;
                                position = new Vector3(x + 1.147f, 0f, z*2.185f + 0.1f);
                                cell = Instantiate(prefabToUse, position, Quaternion.Euler(0, 180, 0));
                                cell.transform.SetParent(transform);

                                myBool = true;
                            }

                            else if(x==0 && z == totalSize - 1){
                                prefabToUse = cornerBottomLeft;
                                position = new Vector3(x + 2.2f, 0f, z*4.1f);
                                cell = Instantiate(prefabToUse, position, Quaternion.Euler(0, 90, 0));
                                cell.transform.SetParent(transform);
                                myBool = true;
                            }
                            
                            else if(x== totalSize - 1 && z == 0){
                                position = new Vector3(x*3.917f , 0f, z * 0.9f);
                                prefabToUse = cornerTopRight;
                                cell = Instantiate(prefabToUse, position, Quaternion.Euler(0, -90, 0));
                                cell.transform.SetParent(transform);
                                myBool = true;
                                
                            }

                            else if(x== totalSize - 1 && z == totalSize - 1){
                                position = new Vector3(x*3.917f , 0f, z + 1.25f);
                                prefabToUse = cornerBottomRight;
                                cell = Instantiate(prefabToUse, position, Quaternion.Euler(0, -180, 0));
                                cell.transform.SetParent(transform);
                                myBool = true;
                            }

                            else if((x==0) && (z>0 &&  z < totalSize - 1)){
                                position = new Vector3(x +1.1f, 0f, z*4.3f + 1);
                                prefabToUse = outerLayerLeft;
                                cell = Instantiate(prefabToUse, position, Quaternion.identity);
                                cell.transform.SetParent(transform);
                                myBool = true;
                                
                            }
                            else if((x==totalSize - 1) && (z>0 &&  z < totalSize - 1)){
                                position = new Vector3(x *4.511f, 0f, z*4.3f + 1);
                                prefabToUse = outerLayerRight;
                                cell = Instantiate(prefabToUse, position, Quaternion.identity);
                                cell.transform.SetParent(transform);
                                myBool = true;
                            }

                            else if((z==0) && (x>0 &&  x < totalSize - 1)){
                                position = new Vector3(x * 4.315f, 0f, z + 2.4f);
                                prefabToUse = outerLayerTop;
                                cell = Instantiate(prefabToUse, position, Quaternion.Euler(0, -360, 0));
                                cell.transform.SetParent(transform);
                                myBool = true;

                            }
                            else if((z==totalSize - 1) && (x>0 &&  x < totalSize - 1)){
                                prefabToUse = outerLayerBottom;
                                
                            }
                        

                        else
                    {
                        // Use ground prefab for the main field
                        prefabToUse = groundPrefab;
                    }
                        // Use ground prefab for the main field
                    //}

                    // Instantiate the appropriate prefab
                    if(!myBool){
                        cell = Instantiate(prefabToUse, position, Quaternion.identity);
                        cell.transform.SetParent(transform);
                    }
                    
                }
            }
        }
    }
