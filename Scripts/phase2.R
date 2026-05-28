# Proyecto Final - GeoAI | Embalse de Tominé / Sesquile
# Fase 2 - Los 5 Modelos + Comparación Final
# Modelos: Decision Tree (DT), KNN, SVM, ANN, Naive Bayes (NB)

# Carga de paquetes necesarios para el análisis
library(tidyverse)
library(caret)
library(rpart)
library(rpart.plot)
library(e1071)
library(nnet)
library(kernlab)
library(ggplot2)
library(gridExtra)
library(reshape2)

# Semilla para reproducibilidad de resultados
set.seed(42)


# CARGA Y PREPARACIÓN DEL DATASET

# Lectura del archivo TSV y limpieza de fila duplicada de encabezado
ruta_tsv <- "C:/Users/Juanf/Downloads/datos_entrenamiento_sesquile_finales.tsv"

df <- read_tsv(ruta_tsv, show_col_types = FALSE) %>%
  filter(clase != "clase") %>%
  mutate(clase = as.factor(clase))

# Definición de las bandas espectrales que se usarán como variables predictoras
BANDAS <- c("B02", "B03", "B04", "B05", "B06", "B08", "B11", "B12")

# Resumen del dataset cargado
cat("Dataset cargado\n")
cat(sprintf("Registros totales : %d\n", nrow(df)))
cat(sprintf("Clases            : %s\n", paste(levels(df$clase), collapse = ", ")))
cat("\nDistribución de clases:\n")
print(table(df$clase))
cat("\nProporción (%):\n")
print(round(prop.table(table(df$clase)) * 100, 2))


# DIVISIÓN TRAIN / TEST Y NORMALIZACIÓN

# Partición estratificada 80/20 para mantener la proporción de clases en ambos conjuntos
idx_train <- createDataPartition(df$clase, p = 0.80, list = FALSE)
train_df  <- df[ idx_train, ]
test_df   <- df[-idx_train, ]

cat(sprintf("\nTrain: %d | Test: %d\n", nrow(train_df), nrow(test_df)))

# Normalización center+scale calculada sobre train y aplicada a test
# Es obligatoria para KNN, SVM y ANN que son sensibles a la escala de los datos
pre_proc     <- preProcess(train_df[, BANDAS], method = c("center", "scale"))
train_scaled <- cbind(predict(pre_proc, train_df[, BANDAS]), clase = train_df$clase)
test_scaled  <- cbind(predict(pre_proc, test_df[,  BANDAS]), clase = test_df$clase)

# Cálculo de pesos inversamente proporcionales a la frecuencia de cada clase
# Compensa el desbalance severo de la clase 'urbana' (solo 27 registros)
freq_clases  <- table(train_df$clase)
pesos_clases <- as.numeric(1 / freq_clases[levels(train_df$clase)])
names(pesos_clases) <- levels(train_df$clase)

# Configuración del control de entrenamiento compartido por todos los modelos
# 5-fold CV garantiza una evaluación robusta de los hiperparámetros
# Upsampling automático para compensar el desbalance de clases durante el CV
ctrl <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  savePredictions = "final",
  verboseIter     = FALSE,
  sampling        = "up"
)

# Lista donde se almacenan los resultados de cada modelo para la comparación final
resultados <- list()

# Función auxiliar que extrae OA, Kappa, F1 por clase y F1 ponderado de cualquier modelo
extraer_metricas <- function(nombre, y_pred, y_real) {
  cm    <- confusionMatrix(y_pred, y_real)
  oa    <- as.numeric(cm$overall["Accuracy"])
  kappa <- as.numeric(cm$overall["Kappa"])
  
  por_clase <- as.data.frame(cm$byClass[, c("Precision", "Recall", "F1")])
  por_clase$Clase <- gsub("Class: ", "", rownames(por_clase))
  
  # F1 ponderado por el soporte real de cada clase en el conjunto de prueba
  weighted_f1 <- weighted.mean(por_clase$F1, table(y_real), na.rm = TRUE)
  
  list(
    nombre    = nombre,
    oa        = oa,
    kappa     = kappa,
    f1_w      = weighted_f1,
    cm        = cm,
    por_clase = por_clase
  )
}


# MODELO 1 - ÁRBOL DE DECISIÓN (Decision Tree)

# El árbol de decisión divide el espacio espectral mediante umbrales en las bandas.
# El parámetro cp (complexity parameter) controla cuánto puede crecer el árbol:
# valores pequeños permiten árboles más profundos y específicos.
# Se aplican pesos por clase para que el árbol preste más atención a 'urbana'.

grid_dt <- expand.grid(cp = c(0.0001, 0.001, 0.005, 0.01, 0.05, 0.1))

# Búsqueda del cp óptimo mediante GridSearch con 5-fold CV
modelo_dt_cv <- train(
  x = train_df[, BANDAS], y = train_df$clase,
  method    = "rpart",
  trControl = ctrl,
  tuneGrid  = grid_dt,
  metric    = "Accuracy"
)

# Entrenamiento del árbol final con el cp óptimo encontrado y pesos por clase
arbol_final <- rpart(
  clase ~ B02 + B03 + B04 + B05 + B06 + B08 + B11 + B12,
  data    = train_df,
  method  = "class",
  weights = pesos_clases[as.character(train_df$clase)],
  control = rpart.control(cp = modelo_dt_cv$bestTune$cp, minsplit = 5, maxdepth = 15)
)

# Predicción sobre el conjunto de prueba y extracción de métricas
pred_dt <- predict(arbol_final, newdata = test_df[, BANDAS], type = "class")
res_dt  <- extraer_metricas("Decision Tree", pred_dt, test_df$clase)
resultados[["DT"]] <- res_dt

cat(sprintf("\nDecision Tree | OA = %.4f | Kappa = %.4f | F1-w = %.4f | cp = %.4f\n",
            res_dt$oa, res_dt$kappa, res_dt$f1_w, modelo_dt_cv$bestTune$cp))


# MODELO 2 - K-VECINOS MÁS CERCANOS (KNN)

# KNN clasifica cada píxel según la clase más frecuente entre sus k vecinos más cercanos
# en el espacio espectral de 8 dimensiones. Es un método no paramétrico, lo que significa
# que no aprende una función explícita, sino que memoriza los datos de entrenamiento.
# Se usa el conjunto normalizado porque KNN es muy sensible a la escala de las variables.

grid_knn <- expand.grid(k = c(3, 5, 7, 9, 11, 15, 21, 31))

# Búsqueda del k óptimo mediante GridSearch con 5-fold CV sobre datos normalizados
modelo_knn <- train(
  x = train_scaled[, BANDAS], y = train_scaled$clase,
  method    = "knn",
  trControl = ctrl,
  tuneGrid  = grid_knn,
  metric    = "Accuracy"
)

# Predicción sobre el conjunto de prueba normalizado y extracción de métricas
pred_knn <- predict(modelo_knn, newdata = test_scaled[, BANDAS])
res_knn  <- extraer_metricas("KNN", pred_knn, test_df$clase)
resultados[["KNN"]] <- res_knn

cat(sprintf("KNN           | OA = %.4f | Kappa = %.4f | F1-w = %.4f | k = %d\n",
            res_knn$oa, res_knn$kappa, res_knn$f1_w, modelo_knn$bestTune$k))


# MODELO 3 - MÁQUINA DE SOPORTE VECTORIAL (SVM)

# SVM busca el hiperplano óptimo que maximiza el margen entre clases en el espacio espectral.
# Se usa el kernel RBF (Radial Basis Function) que proyecta los datos a un espacio
# de mayor dimensión donde las clases son linealmente separables.
# C controla la tolerancia al error: valores altos penalizan más los errores de clasificación.
# Sigma define el radio de influencia de cada punto de entrenamiento.

grid_svm <- expand.grid(
  C     = c(0.1, 1, 10),
  sigma = c(0.01, 0.1, 1)
)

# Búsqueda de C y sigma óptimos mediante GridSearch con 5-fold CV
modelo_svm <- train(
  x = train_scaled[, BANDAS], y = train_scaled$clase,
  method    = "svmRadial",
  trControl = ctrl,
  tuneGrid  = grid_svm,
  metric    = "Accuracy"
)

# Predicción sobre el conjunto de prueba normalizado y extracción de métricas
pred_svm <- predict(modelo_svm, newdata = test_scaled[, BANDAS])
res_svm  <- extraer_metricas("SVM", pred_svm, test_df$clase)
resultados[["SVM"]] <- res_svm

cat(sprintf("SVM           | OA = %.4f | Kappa = %.4f | F1-w = %.4f | C = %.1f | sigma = %.2f\n",
            res_svm$oa, res_svm$kappa, res_svm$f1_w,
            modelo_svm$bestTune$C, modelo_svm$bestTune$sigma))


# MODELO 4 - RED NEURONAL ARTIFICIAL (ANN)

# La red neuronal aprende relaciones no lineales entre las bandas espectrales y las clases.
# Se usa una arquitectura de una capa oculta (nnet). El parámetro size define el número
# de neuronas en esa capa, y decay es la regularización L2 que penaliza pesos grandes
# para evitar sobreajuste. Se entrena con un máximo de 300 iteraciones de optimización.

grid_ann <- expand.grid(
  size  = c(5, 10, 20),
  decay = c(0.001, 0.01, 0.1)
)

# Búsqueda de size y decay óptimos mediante GridSearch con 5-fold CV
modelo_ann <- train(
  x = train_scaled[, BANDAS], y = train_scaled$clase,
  method    = "nnet",
  trControl = ctrl,
  tuneGrid  = grid_ann,
  metric    = "Accuracy",
  maxit     = 300,
  trace     = FALSE
)

# Predicción sobre el conjunto de prueba normalizado y extracción de métricas
pred_ann <- predict(modelo_ann, newdata = test_scaled[, BANDAS])
res_ann  <- extraer_metricas("ANN", pred_ann, test_df$clase)
resultados[["ANN"]] <- res_ann

cat(sprintf("ANN           | OA = %.4f | Kappa = %.4f | F1-w = %.4f | neuronas = %d | decay = %.3f\n",
            res_ann$oa, res_ann$kappa, res_ann$f1_w,
            modelo_ann$bestTune$size, modelo_ann$bestTune$decay))


# MODELO 5 - NAIVE BAYES (NB)

# Naive Bayes es un clasificador probabilístico basado en el Teorema de Bayes.
# Asume independencia condicional entre las bandas espectrales (supuesto "naive"),
# lo que simplifica el cálculo pero puede ser una limitación dado que bandas como
# B03 y B04 están correlacionadas. usekernel = TRUE estima la densidad de probabilidad
# de forma no paramétrica en lugar de asumir distribución gaussiana.
# fL es la corrección de Laplace para evitar probabilidades cero.

grid_nb <- expand.grid(
  usekernel = c(TRUE, FALSE),
  fL        = c(0, 0.5, 1),
  adjust    = c(0.5, 1, 2)
)

# Búsqueda de hiperparámetros óptimos mediante GridSearch con 5-fold CV
modelo_nb <- train(
  x = train_df[, BANDAS], y = train_df$clase,
  method    = "nb",
  trControl = ctrl,
  tuneGrid  = grid_nb,
  metric    = "Accuracy"
)

# Predicción sobre el conjunto de prueba y extracción de métricas
pred_nb <- predict(modelo_nb, newdata = test_df[, BANDAS])
res_nb  <- extraer_metricas("Naive Bayes", pred_nb, test_df$clase)
resultados[["NB"]] <- res_nb

cat(sprintf("Naive Bayes   | OA = %.4f | Kappa = %.4f | F1-w = %.4f\n",
            res_nb$oa, res_nb$kappa, res_nb$f1_w))


# TABLA COMPARATIVA FINAL

# Construcción de la tabla resumen ordenada de mayor a menor OA
comparacion <- map_dfr(resultados, function(r) {
  tibble(
    Modelo   = r$nombre,
    OA       = round(r$oa    * 100, 2),
    Kappa    = round(r$kappa,  4),
    F1_Score = round(r$f1_w,   4)
  )
}) %>% arrange(desc(OA))

comparacion$Mejor <- ifelse(comparacion$OA == max(comparacion$OA), "<-- MEJOR", "")

print(comparacion, row.names = FALSE)

mejor_modelo <- comparacion$Modelo[1]
cat(sprintf("\nModelo recomendado para inferencia: %s\n", mejor_modelo))


# GRÁFICAS

# Gráfica de barras comparando OA, Kappa y F1 de los 5 modelos
comp_long <- comparacion %>%
  select(Modelo, OA, Kappa, F1_Score) %>%
  mutate(Kappa = Kappa * 100, F1_Score = F1_Score * 100) %>%
  pivot_longer(cols = c(OA, Kappa, F1_Score), names_to = "Metrica", values_to = "Valor")

png("comparacion_modelos.png", width = 900, height = 550, res = 130)
ggplot(comp_long, aes(x = reorder(Modelo, Valor), y = Valor, fill = Metrica)) +
  geom_col(position = "dodge", width = 0.7) +
  geom_text(aes(label = sprintf("%.1f", Valor)),
            position = position_dodge(width = 0.7),
            vjust = -0.4, size = 3) +
  scale_fill_manual(values = c("OA" = "#1565C0", "Kappa" = "#2E7D32", "F1_Score" = "#E65100")) +
  labs(title    = "Comparación de Modelos — Cobertura del Suelo (Sesquile)",
       subtitle = "Overall Accuracy, Kappa y F1-Score ponderado (%)",
       x = NULL, y = "Valor (%)", fill = "Métrica") +
  ylim(0, 115) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top")
dev.off()
cat("Figura guardada: comparacion_modelos.png\n")

# Matrices de confusión de los 5 modelos juntas en una sola imagen
# Cada celda muestra cuántos píxeles de la clase real fueron predichos como cada clase
png("todas_confusion_matrices.png", width = 1600, height = 700, res = 130)
plots_cm <- lapply(seq_along(resultados), function(i) {
  r     <- resultados[[i]]
  cm_df <- as.data.frame(r$cm$table)
  names(cm_df) <- c("Predicho", "Real", "Freq")
  ggplot(cm_df, aes(x = Real, y = Predicho, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), size = 3.5, fontface = "bold") +
    scale_fill_gradient(low = "#E3F2FD", high = "#1565C0") +
    labs(title = sprintf("%s\nOA=%.1f%% | K=%.3f", r$nombre, r$oa * 100, r$kappa),
         x = "Real", y = "Predicho") +
    theme_minimal(base_size = 9) +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 30, hjust = 1))
})
gridExtra::grid.arrange(grobs = plots_cm, nrow = 1)
dev.off()
cat("Figura guardada: todas_confusion_matrices.png\n")

# Heatmap de F1-Score por clase y modelo para identificar fortalezas y debilidades específicas
f1_tabla <- map_dfr(resultados, function(r) {
  r$por_clase %>%
    mutate(Modelo = r$nombre,
           Clase  = gsub("Class: ", "", Clase),
           F1     = round(F1, 3)) %>%
    select(Modelo, Clase, F1)
})

png("f1_por_clase_y_modelo.png", width = 800, height = 500, res = 130)
ggplot(f1_tabla, aes(x = Modelo, y = Clase, fill = F1)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.3f", F1)), size = 3.5, fontface = "bold") +
  scale_fill_gradient2(low = "#B71C1C", mid = "#FFF9C4", high = "#1B5E20",
                       midpoint = 0.75, limits = c(0, 1), na.value = "grey80") +
  labs(title = "F1-Score por Clase y Modelo", x = NULL, y = NULL, fill = "F1") +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))
dev.off()
cat("Figura guardada: f1_por_clase_y_modelo.png\n")

# Visualización de la estructura del árbol de decisión óptimo
# Muestra las bandas y umbrales usados en cada partición, y la clase predicha en cada hoja
png("DT_tree_visualization.png", width = 1400, height = 700, res = 120)
rpart.plot(arbol_final, type = 4, extra = 104, fallen.leaves = TRUE,
           main = "Árbol de Decisión — Cobertura del Suelo (Sesquile)", cex = 0.7)
dev.off()
cat("Figura guardada: DT_tree_visualization.png\n")

# Importancia relativa de cada banda según el criterio Gini del árbol de decisión
# Una banda con mayor importancia contribuye más a las particiones del árbol
imp <- arbol_final$variable.importance
if (!is.null(imp) && length(imp) > 0) {
  imp_df <- data.frame(
    Banda       = names(imp),
    Importancia = as.numeric(imp) / sum(imp)
  )
  imp_df <- imp_df[order(imp_df$Importancia, decreasing = FALSE), ]
  
  png("DT_feature_importance.png", width = 700, height = 450, res = 120)
  print(
    ggplot(imp_df, aes(x = factor(Banda, levels = Banda),
                       y = Importancia, fill = Importancia)) +
      geom_col(show.legend = FALSE) +
      geom_text(aes(label = sprintf("%.3f", Importancia)),
                hjust = -0.1, size = 3.5, color = "black") +
      scale_fill_gradient(low = "#90CAF9", high = "#1565C0") +
      coord_flip() +
      ylim(0, max(imp_df$Importancia) * 1.2) +
      labs(title = "Importancia de Bandas — DT",
           x = "Banda", y = "Importancia Relativa") +
      theme_minimal(base_size = 12)
  )
  dev.off()
  cat("Figura guardada: DT_feature_importance.png\n")
}

# Curva de accuracy en función de K para el modelo KNN evaluada por validación cruzada
# Permite visualizar cómo el rendimiento decrece al aumentar el número de vecinos
png("KNN_accuracy_vs_k.png", width = 700, height = 430, res = 120)
ggplot(modelo_knn$results, aes(x = k, y = Accuracy)) +
  geom_line(color = "#2E7D32", linewidth = 1) +
  geom_point(color = "#2E7D32", size = 3) +
  geom_vline(xintercept = modelo_knn$bestTune$k, linetype = "dashed", color = "red") +
  labs(title = "KNN — Accuracy vs K (CV)", x = "K", y = "Accuracy") +
  theme_minimal(base_size = 12)
dev.off()
cat("Figura guardada: KNN_accuracy_vs_k.png\n")


# EXPORTAR RESULTADOS

# Tabla comparativa general de los 5 modelos
write_csv(comparacion, "comparacion_5_modelos.csv")
cat("\nTabla exportada: comparacion_5_modelos.csv\n")

# Tabla con métricas detalladas por clase para cada modelo
metricas_detalle <- map_dfr(resultados, function(r) {
  r$por_clase %>%
    mutate(Modelo = r$nombre,
           OA     = round(r$oa,    4),
           Kappa  = round(r$kappa, 4)) %>%
    select(Modelo, Clase, Precision, Recall, F1, OA, Kappa) %>%
    mutate(across(where(is.numeric), ~ round(.x, 4)))
})
write_csv(metricas_detalle, "metricas_detalle_5_modelos.csv")
cat("Tabla exportada: metricas_detalle_5_modelos.csv\n")


# RESUMEN EJECUTIVO FINAL

cat(sprintf("\nDataset           : %d registros | %d clases\n", nrow(df), nlevels(df$clase)))
cat(sprintf("Train/Test        : %d / %d (80/20 estratificado)\n", nrow(train_df), nrow(test_df)))
cat("\n")
for (nm in names(resultados)) {
  r <- resultados[[nm]]
  cat(sprintf("  %-14s  OA = %5.2f%%  Kappa = %.4f  F1 = %.4f\n",
              r$nombre, r$oa * 100, r$kappa, r$f1_w))
}
cat(sprintf("\n  Mejor modelo  : %s\n", mejor_modelo))