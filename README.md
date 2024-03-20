# Treball de Fi de Màster

## Replicar resultats
Primer que res, s'intentarà replicar els resultats presentats
en el paper _[UNITOPATHO, A LABELED HISTOPATHOLOGICAL DATASET FOR COLORECTAL POLYPS CLASSIFICATION AND ADENOMA DYSPLASIA GRADING](https://ieeexplore.ieee.org/document/9506198)_.
- Data augmentation
- Entrenament d'una xarxa neuronal convolucional. A partir d'un model ImageNet ResNet-18 preentrenat amb SGD durant 50 epochs amb un learning rate de 0.01, que disminueix un 10% cada 20 epochs.


### Dades
| Top label | Top label id |
| :-: | :-: |
| HP     | 0 |
| NORM   | 1 |
| TA.HG  | 2 |
| TA.LG  | 3 |
| TVA.HG | 4 |
| TVA.LG | 5 |

| Type | Type id |
| :-: | :-: |
| HP   | 0 |
| NORM | 1 |
| TA   | 2 |
| TVA  | 3 |

| Grade | Grade id |
| :-: | :-: |
| `null` (for HP and NORM type) | -1 |
| LG | 0 |
| HG | 1 |

Amb el paràmetre `--target` es pot decidir sobre quina categoria (`top_label`, `type` o `grade`) es
desitja entrenar la xarxa neuronal.

Quan s'utilitza la categoria `grade`, es pot decidir si es desitja treballar sobre `ta`, `tva`, `norm` (?) o `both`.
