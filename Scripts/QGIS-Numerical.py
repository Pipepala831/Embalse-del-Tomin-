import os
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.mask import mask

# 1. Rutas de tus archivos (Usamos 'r' al inicio para evitar errores de barras '\' en Windows)
ruta_geojson = r"..\ShapeFiles\poligonos_entrenamiento.geojson"
carpeta_bandas = r"..\ano_1"

# Definimos las 8 bandas que descargaste
bandas_nombres = ["B02", "B03", "B04", "B05", "B06", "B08", "B11", "B12"]

# 2. Cargar los polígonos de entrenamiento usando GeoPandas
print("Cargando polígonos de entrenamiento...")
gdf_poligonos = gpd.read_file(ruta_geojson)

# Lista para almacenar los datos de cada píxel extraído
datos_pixeles = []

# --- BUSCADOR AUTOMÁTICO DE LA BANDA BASE ---
# Esto escanea la carpeta y encuentra el archivo de la banda B02 sin importar las fechas del nombre
archivos_en_carpeta = os.listdir(carpeta_bandas)
archivo_b2_real = [f for f in archivos_en_carpeta if "B02" in f and f.lower().endswith(('.tif', '.tiff'))][0]
ruta_b2 = os.path.join(carpeta_bandas, archivo_b2_real)

print(f"Banda base de referencia detectada: {archivo_b2_real}")

with rasterio.open(ruta_b2) as src_ref:
    crs_referencia = src_ref.crs
    transformacion_afin = src_ref.transform

# Asegurar que los polígonos usen exactamente el mismo sistema de coordenadas (CRS) que el ráster
if gdf_poligonos.crs != crs_referencia:
    print(f"Reproyectando polígonos de {gdf_poligonos.crs} a {crs_referencia}...")
    gdf_poligonos = gdf_poligonos.to_crs(crs_referencia)

# 3. Procesar banda por banda
# Para cumplir la alineación de píxeles, leeremos las matrices completas
matrices_bandas = {}

print("Leyendo y verificando alineación espacial de las 8 bandas...")
for b in bandas_nombres:
    # Buscar dinámicamente el archivo correspondiente a cada banda
    archivo_banda = [f for f in archivos_en_carpeta if b in f and f.lower().endswith(('.tif', '.tiff'))][0]
    ruta_completa = os.path.join(carpeta_bandas, archivo_banda)
    
    with rasterio.open(ruta_completa) as src:
        # Nota de alineación: Si las bandas de 20m (B05, B06, B11, B12) no coinciden en tamaño 
        # con las de 10m, rasterio las re-muestrea (resample) automáticamente al tamaño de la de referencia
        if src.shape != src_ref.shape:
            matrices_bandas[b] = src.read(
                1, 
                out_shape=(src_ref.height, src_ref.width),
                resampling=rasterio.enums.Resampling.bilinear
            )
        else:
            matrices_bandas[b] = src.read(1)

print("Extrayendo firmas espectrales dentro de los polígonos...")

# 4. Iterar sobre cada polígono dibujado en QGIS para extraer los píxeles internos
for index, fila in gdf_poligonos.iterrows():
    geometria = fila['geometry']
    nombre_clase = fila['clase']
    
    # Usamos rasterio.mask para recortar la matriz usando el polígono actual
    with rasterio.open(ruta_b2) as src:
        mascara_recorte, transformacion_recortada = mask(src, [geometria], crop=True, nodata=-9999)
        mascara_indices = mascara_recorte[0] != -9999
        
    # Si el polígono encerró píxeles válidos
    if np.any(mascara_indices):
        # Obtener los índices de fila y columna (I, J) en la imagen original
        with rasterio.open(ruta_b2) as src:
            # Re-calculamos la máscara sobre la geometría completa sin recortar la matriz
            mascara_completa, _ = mask(src, [geometria], crop=False, nodata=-9999)
            filas_idx, columnas_idx = np.where(mascara_completa[0] != -9999)
            
        for f, c in zip(filas_idx, columnas_idx):
            # Transformación Afín: Convertir la posición del píxel (fila, columna) a coordenadas proyectadas (X, Y)
            x, y = rasterio.transform.xy(transformacion_afin, f, c)
            
            # Convertir coordenadas planas (X,Y) a coordenadas geográficas (Latitud, Longitud)
            longitud, latitud = rasterio.warp.transform(crs_referencia, 'EPSG:4326', [x], [y])
            
            # Diccionario base con la ubicación y la etiqueta
            pixel_dict = {
                "Latitude": latitud[0],
                "Longitude": longitud[0]
            }
            
            # Agregar el valor numérico de reflectancia de las 8 bandas para ese píxel
            for b in bandas_nombres:
                pixel_dict[b] = matrices_bandas[b][f, c]
                
            # Agregar la etiqueta de cobertura (ej: Agua, Vegetacion)
            pixel_dict["clase"] = nombre_clase
            
            datos_pixeles.append(pixel_dict)

# 5. Convertir a un DataFrame de Pandas y exportar a formato TSV
df_final = pd.DataFrame(datos_pixeles)

# Eliminar posibles valores nulos o lecturas erróneas fuera del rango del satélite
df_final = df_final.dropna()

# Guardar en la misma carpeta de scripts de forma relativa
output_path = "datos_entrenamiento_sesquile.tsv"
df_final.to_csv(output_path, sep='\t', index=False)

print(f"\n¡Fase 1 Completada con éxito! Archivo guardado en: {output_path}")
print(f"Total de píxeles extraídos para entrenamiento: {len(df_final)}")
print("\nMuestra de las primeras filas de la tabla generada:")
print(df_final.head())