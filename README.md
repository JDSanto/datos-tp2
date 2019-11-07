# Datos - TP2

Links:
- [Set de entrenamiento](https://metadata.fundacionsadosky.org.ar/media/navent/train.csv). 
- [Set de testeo](https://metadata.fundacionsadosky.org.ar/media/navent/test.csv). 
- Resultados deben tener [este formato](https://metadata.fundacionsadosky.org.ar/media/navent/ejemploRespuesta.csv).
- [Tutorial de Navent](https://metadata.fundacionsadosky.org.ar/media/navent/metaDataNavent.ipynb)

## Informe:
```bash
$ make generarInforme
```

## Requerimientos:
```bash
$ pip install -r requirements.txt
```
## Subir a Kaggle:

#### Configuración
1. En la consola de Firebase, ir a [Cuentas de servicio](https://console.firebase.google.com/u/0/project/datos-tp2-20192c/settings/serviceaccounts/adminsdk)
2. Descargar la clave nueva: `Generar nueva clave privada`
3. Copiarla en la raiz del proyecto
4. Cambiarle el nombre a: `serviceAccountKey.json`
5. Ir a sección `account` del perfil de Kaggle
6. En la parte de API: `Create New API Token`
7. Copiar el archivo descargado a la carpeta `~/.kaggle`. Ej: `mkdir ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle`
8. Correr `chmod 600 ~/.kaggle/kaggle.json` para evitar warnings

#### Subir:
```bash
$ python kaggle_pipeline.py <RUTA AL CSV GENERADO> <DESCRIPCIÓN>
```

Por ejemplo:
```bash
$ python kaggle_pipeline.py ./data/respuesta.csv "Con este seguro ganamos"
```
