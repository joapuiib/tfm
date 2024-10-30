# Crear el codebook
```
kmeans = new KMeans(10)

for image in images:
    Mat image = imread(image)

    # Investigar les diferents codificacions de color
    # Opcions de lesctura: imread

    for pixel in image:
        pixel pertany a std:vector[3] 0-255
        kmeans.add(pixel)

guardar el codebook
```

# Transformar imatge
```
load_codebook()

imatge_transformada = [][]

Mat pixels = imread(image)
for pixel in pixels:
    pixel pertany a std:vector[3] 0-255
    
    classe = kmeans.transform(pixel)
    # Transformar a escala de grisos (classe / K)

    imatge_transformada.append(pixel)

```
