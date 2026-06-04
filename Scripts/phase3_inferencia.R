# Proyecto Final - GeoAI | Embalse de Tominé / Sesquile
# Fase 3 - Inferencia sobre el raster completo
# Aplica el modelo ANN entrenado sobre todos los píxeles de la imagen Sentinel-2
# y exporta el mapa de coberturas como GeoTIFF georreferenciado

# Paquetes necesarios
install.packages("terra")
library(tidyverse)
library(caret)
library(nnet)
library(terra)


# RUTAS — ajustar según la ubicación de los archivos en tu equipo

carpeta_bandas <- "C:/Users/Juanf/Downloads/Embalse-del-Tomin--main/Embalse-del-Tomin--main/Ano_1"
ruta_salida    <- "C:/Users/Juanf/Downloads/mapa_coberturas_tomine_2022.tif"

# Nombres de las 8 bandas (mismo orden que el dataset de entrenamiento)
BANDAS <- c("B02", "B03", "B04", "B05", "B06", "B08", "B11", "B12")

# Mapeo de clases a códigos enteros para el raster de salida
# (orden alfabético = mismo orden que caret codifica los factores)
codigos_clase <- c("agua" = 1, "nube" = 2, "urbana" = 3, "vegetacion" = 4)

cat("Fase 3 — Inferencia sobre raster completo\n")
cat(sprintf("Carpeta de bandas: %s\n", carpeta_bandas))


# CARGA DE LAS BANDAS RASTER

cat("\nCargando bandas Sentinel-2...\n")

# Detectar automáticamente los archivos tiff por nombre de banda
archivos_banda <- list()
for (b in BANDAS) {
  patron   <- paste0("_", b, "_")
  archivos <- list.files(carpeta_bandas, pattern = patron, full.names = TRUE)
  archivos <- archivos[grepl("\\.tiff?$", archivos, ignore.case = TRUE)]
  if (length(archivos) == 0) {
    stop(sprintf("No se encontró archivo para la banda %s en %s", b, carpeta_bandas))
  }
  archivos_banda[[b]] <- archivos[1]
  cat(sprintf("  %s -> %s\n", b, basename(archivos_banda[[b]])))
}

# Cargar la banda de referencia (B02, 10m) para obtener CRS y extensión
raster_ref <- rast(archivos_banda[["B02"]])
cat(sprintf("\nDimensiones de referencia (B02): %d filas x %d columnas\n",
            nrow(raster_ref), ncol(raster_ref)))
cat(sprintf("CRS: %s\n", crs(raster_ref, describe = TRUE)$name))
cat(sprintf("Total píxeles: %d\n", nrow(raster_ref) * ncol(raster_ref)))


# CONSTRUCCIÓN DEL STACK MULTIBANDA

cat("\nRemuestreando bandas de 20m a 10m y construyendo stack...\n")

stack_lista <- list()
for (b in BANDAS) {
  r <- rast(archivos_banda[[b]])
  # Remuestrear a la cuadrícula de referencia si la resolución difiere
  if (!compareGeom(r, raster_ref, stopOnError = FALSE)) {
    r <- resample(r, raster_ref, method = "bilinear")
    cat(sprintf("  %s remuestreada a 10m\n", b))
  }
  stack_lista[[b]] <- r
}

# Apilar todas las bandas en un solo raster multibanda
stack_bandas <- rast(stack_lista)
names(stack_bandas) <- BANDAS
cat("Stack multibanda construido correctamente.\n")


# CONVERSIÓN A MATRIZ PARA PREDICCIÓN

cat("\nExtrayendo valores de píxeles...\n")

# Convertir el stack a una matriz: cada fila es un píxel, cada columna una banda
# na.rm = FALSE para conservar la posición espacial de cada píxel
mat_pixeles <- as.data.frame(stack_bandas, xy = FALSE, na.rm = FALSE)
names(mat_pixeles) <- BANDAS

cat(sprintf("Matriz de píxeles: %d filas x %d columnas\n",
            nrow(mat_pixeles), ncol(mat_pixeles)))

# Identificar píxeles con datos válidos (no NA)
# Los píxeles fuera de la escena tienen NA en todas las bandas
mask_validos <- complete.cases(mat_pixeles)
cat(sprintf("Píxeles válidos: %d | Píxeles NoData: %d\n",
            sum(mask_validos), sum(!mask_validos)))


# NORMALIZACIÓN

cat("\nAplicando normalización center+scale del entrenamiento...\n")

# Aplicar EXACTAMENTE el mismo pre_proc usado en el entrenamiento
# para garantizar que las escalas sean consistentes
mat_scaled <- mat_pixeles
mat_scaled[mask_validos, ] <- predict(pre_proc, mat_pixeles[mask_validos, ])


# PREDICCIÓN PÍXEL A PÍXEL

cat("\nEjecutando inferencia con modelo ANN...\n")
cat("(Esto puede tardar varios minutos dependiendo del tamaño del raster)\n")

# Inicializar vector de predicciones con NA
predicciones_clase  <- rep(NA_character_, nrow(mat_pixeles))

# Predecir solo sobre píxeles válidos para no desperdiciar tiempo en NoData
predicciones_clase[mask_validos] <- as.character(
  predict(modelo_ann, newdata = mat_scaled[mask_validos, ])
)

cat(sprintf("Predicción completada. Distribución de clases predichas:\n"))
print(table(predicciones_clase, useNA = "ifany"))


# CODIFICACIÓN A ENTEROS

cat("\nCodificando clases a valores enteros...\n")
# agua=1, nube=2, urbana=3, vegetacion=4
predicciones_int <- codigos_clase[predicciones_clase]
predicciones_int[is.na(predicciones_clase)] <- NA

cat("Leyenda del raster de salida:\n")
for (cls in names(codigos_clase)) {
  n <- sum(predicciones_clase == cls, na.rm = TRUE)
  cat(sprintf("  %d = %-12s : %d píxeles\n", codigos_clase[cls], cls, n))
}


# CONSTRUCCIÓN DEL RASTER DE SALIDA

cat("\nConstruyendo raster de salida...\n")

# Crear raster con la misma geometría que la referencia
raster_salida <- rast(raster_ref)
values(raster_salida) <- predicciones_int

# Asignar tabla de categorías para que QGIS reconozca las clases
levels(raster_salida) <- data.frame(
  value = unname(codigos_clase),
  label = names(codigos_clase)
)
names(raster_salida) <- "cobertura"


# EXPORTAR GEOTIFF

cat(sprintf("\nExportando GeoTIFF: %s\n", ruta_salida))

writeRaster(
  raster_salida,
  filename  = ruta_salida,
  datatype  = "INT1U",     # enteros sin signo de 1 byte (0-255), suficiente para 4 clases
  overwrite = TRUE
)

cat("GeoTIFF exportado correctamente.\n")


# CÁLCULO DE ÁREAS POR CLASE

cat("\nCalculando áreas por clase...\n")

# Resolución espacial en metros (10m x 10m = 100 m² por píxel)
res_m   <- res(raster_salida)
area_px <- res_m[1] * res_m[2]   # m² por píxel

tabla_areas <- data.frame(
  Clase    = names(codigos_clase),
  Codigo   = unname(codigos_clase),
  Pixeles  = sapply(names(codigos_clase), function(cls) {
               sum(predicciones_clase == cls, na.rm = TRUE)
             })
) %>%
  mutate(
    Area_ha  = round(Pixeles * area_px / 10000, 2),
    Area_km2 = round(Pixeles * area_px / 1e6,   4),
    Pct      = round(Pixeles / sum(Pixeles) * 100, 2)
  ) %>%
  arrange(desc(Pixeles))

cat("\nTabla de áreas por clase de cobertura:\n")
print(tabla_areas, row.names = FALSE)

# Exportar tabla de áreas como CSV
ruta_areas <- gsub("\\.tif$", "_areas.csv", ruta_salida)
write_csv(tabla_areas, ruta_areas)
cat(sprintf("\nTabla de áreas exportada: %s\n", ruta_areas))


# INSTRUCCIONES PARA QGIS

cat("\n--- INSTRUCCIONES PARA VISUALIZAR EN QGIS ---\n")
cat("1. Abrir QGIS y cargar el archivo:\n")
cat(sprintf("   %s\n", ruta_salida))
cat("2. Click derecho en la capa > Propiedades > Simbología\n")
cat("3. Seleccionar 'Valores únicos' (Paletted/Unique values)\n")
cat("4. Asignar colores:\n")
cat("   1 = agua       -> Azul   (#1565C0)\n")
cat("   2 = nube       -> Blanco (#F5F5F5)\n")
cat("   3 = urbana     -> Rojo   (#D84315)\n")
cat("   4 = vegetacion -> Verde  (#2E7D32)\n")
cat("5. Aplicar y Aceptar\n")

