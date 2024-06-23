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

### Resultats replicació

#### Entrenament per tipus (`type`)
|           | 800    | Paper 800 | 7000 | 7000_1812 | 7000_224 | Paper 7000 |
| :-:       | :-:    | :-: | :-: | :-: | :-: | :-: |
| NORM      | 0.6541 | 0.66 | * | __0.7653__ | 0.5912     | 0.71 |
| HP        | 0.9002 | 0.92 | * | 0.8022     | __0.8266__ | 0.69 |
| TA        | 0.7230 | 0.66 | * | __0.8177__ | 0.6729     | 0.70 |
| TVA       | 0.7229 | 0.67 | * | __0.8332__ | 0.6685     | 0.76 |
| __Total__ | 0.6378 | ?    | * | __0.7017__ | 0.5440     | ?    |

(*) No es disposa de suficient memòria per l'entrenament 7000 amb les imatges originals.

#### Entrenament totes les classes (`top_label`)
|           | 800    | Paper 800 | 7000 | 7000_1812 | 7000_224 | Paper 7000 |
| :-:       | :-:    | :-: | :-: | :-: | :-: | :-: |
| NORM      | 0.6714 | ?    | * | 0.7620 | 0.8298 | ?    |
| HP        | 0.9254 | ?    | * | 0.7610 | 0.5900 | ?    |
| TA.HG     | 0.5973 | ?    | * | 0.4611 | 0.4678 | ?    |
| TA.LG     | 0.6888 | ?    | * | 0.5585 | 0.6404 | ?    |
| TVA.HG    | 0.5460 | ?    | * | 0.5111 | 0.5711 | ?    |
| TVA.LG    | 0.5468 | ?    | * | 0.5491 | 0.6710 | ?    |
| __Total__ | 0.4448 | 0.45 | * | 0.3445 | 0.3820 | 0.37 |