# Presentacion Academica: Redes Generativas Antagonicas (GAN)

**Proyecto:** Explorador Interactivo de GAN con Implementacion Real en PyTorch  
**Curso:** Inteligencia Artificial Generativa  
**Fecha:** 2026

---

## 1. Objetivos del Proyecto

### Objetivo General
Desarrollar una aplicacion interactiva que permita comprender el funcionamiento de una GAN mediante la implementacion de sus dos componentes principales: el Generador y el Discriminador.

### Objetivos Especificos
- Implementar Generator y Discriminator como redes neuronales MLP independientes
- Filtrar MNIST para entrenar exclusivamente una categoria de digitos (0-9)
- Registrar perdidas, probabilidades, tiempo e imagenes procesadas
- Inspeccionar pesos y sesgos obtenidos de tensores reales
- Persistir checkpoints, metricas CSV y muestras para comparar epocas

---

## 2. Fundamentos Teoricos

### Que es una GAN?

Una **Red Generativa Antagonica (GAN)** es un framework propuesto por Ian Goodfellow en 2014. Consiste en dos redes neuronales que compiten:

1. **El Generador G**: Crea imagenes artificiales a partir de ruido aleatorio
2. **El Discriminador D**: Evalua si una imagen es real o generada

### El Juego Adversario

Durante el entrenamiento, ambas redes mejoran en competencias opuestas:

- **D** aprende a distinguir imagenes reales de las falsas
- **G** aprende a crear imagenes que engañen a D

### Formulas Matematicas

$$L_D = -\mathbb{E}_{x\sim p_{data}}[\log D(x)] - \mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]$$

$$L_G = -\mathbb{E}_{z\sim p_z}[\log D(G(z)))]$$

Donde:
- $x$ = imagenes reales del dataset
- $z$ = vectores de ruido latente
- $D(x)$ = probabilidad de que x sea real
- $G(z)$ = imagen generada a partir de z

---

## 3. Arquitectura Implementada

### Generador (Generator)

Red neuronal que transforma un vector latente de 100 dimensiones en una imagen de 28x28 pixeles:

```
z (100) → Linear(100,256) → LeakyReLU(0.2)
       → Linear(256,512) → LeakyReLU(0.2)
       → Linear(512,1024) → LeakyReLU(0.2)
       → Linear(1024,784) → Tanh
       → reshape → imagen 1×28×28
```

**Caracteristicas:**
- **Dimension latente:** 100 (ruido aleatorio normal N(0,1))
- **Funcion de activacion:** LeakyReLU(0.2) para evitar gradientes muertos
- **Salida:** Tanh normaliza pixeles en [-1, 1]
- **Parametros totales:** ~1,236,288

### Discriminador (Discriminador)

Red que clasifica imagenes como reales o falsas:

```
imagen 1×28×28 → Flatten(784)
              → Linear(784,512) → LeakyReLU(0.2)
              → Linear(512,256) → LeakyReLU(0.2)
              → Linear(256,1) → Logit (sin sigmoid)
```

**Caracteristicas:**
- Devuelve logits (sin sigmoid interna)
- BCEWithLogitsLoss para estabilidad numerica
- **Parametros totales:** ~1,170,817

---

## 4. Parametros de Entrenamiento

| Parametro | Valor |
|-----------|-------|
| Dimension Latente | 100 |
| Batch Size | 128 |
| Learning Rate | 0.0002 |
| Epocas | 30 |
| Optimizador | Adam |
| Betas | (0.5, 0.999) |
| Resolucion | 28x28 |
| Normalizacion | [-1, 1] |

> **Nota importante:** La aplicacion entrena una GAN sobre una categoria especifica de MNIST (por defecto el digito 7). No es un clasificador de digitos.

---

## 5. Dataset MNIST

El conjunto de datos **MNIST** (Modified National Institute of Standards and Technology) contiene:
- 70,000 imagenes de digitos escritos a mano (0-9)
- Resolucion: 28x28 pixeles
- Escala de grises (1 canal)

**Enfoque del proyecto:**
- Descarga automatica mediante torchvision
- Filtrado por categoria seleccionada (0-9)
- Normalizacion a rango [-1, 1]
- Dataset filtrado para digito 7: **6,265 imagenes**

---

## 6. Ciclo de Entrenamiento

El entrenamiento alterna entre dos fases:

### Fase 1: Entrenamiento del Discriminador
1. Obtener lote de imagenes reales
2. Generar imagenes falsas con G (usando detach())
3. Calcular pérdida: BCE(real, 1) + BCE(fake, 0)
4. Actualizar pesos de D

### Fase 2: Entrenamiento del Generador
1. Generar nuevas imagenes falsas
2. Congelar pesos de D
3. Calcular pérdida: BCE(D(G(z)), 1)
4. Actualizar pesos de G

---

## 7. Funciones de Perdida

### Binary Cross Entropy (BCE)

Se utiliza nn.BCEWithLogitsLoss() que combina sigmoid y BCE de forma numericamente estable.

### Perdida del Discriminador
L_D = BCE(D(x), 1) + BCE(D(G(z)), 0)
- Clasificar imagenes reales como 1
- Clasificar imagenes generadas como 0

### Perdida del Generador
L_G = BCE(D(G(z)), 1)
- Intenta que sus imagenes sean clasificadas como 1 (reales)

> **Interpretacion:** Las perdidas de una GAN no necesariamente disminuyen simultáneamente ni deben converger a cero.

---

## 8. Interfaz de la Aplicacion

La aplicacion Streamlit ofrece las siguientes secciones:

1. **Inicio**: Explicacion conceptual del funcionamiento de una GAN
2. **Arquitectura**: Diagramas y detalles de Generator y Discriminator
3. **Configuracion y entrenamiento**: Parametros ajustables, inicio de entrenamiento
4. **Perdidas**: Graficas de evolucion, metricas por epoca
5. **Parametros y pesos**: Inspeccion de tensores reales
6. **Resultados**: Comparacion visual imagenes reales vs generadas
7. **Conclusiones**: Resumen y reflexion sobre el aprendizaje

---

## 9. Visualizacion de Parametros

Para cada capa lineal se muestra:
- Nombre y tipo de capa
- Dimensiones de entrada y salida
- Cantidad de parametros
- Estadisticas: min, max, media, desviacion estandar
- Muestra de valores numericos de los tensores

---

## 10. Persistencia y Reproducibilidad

Cada ejecucion genera:
- config.json: Parametros utilizados
- metrics.csv: Historico de metricas por epoca
- initial_weights.pt: Pesos iniciales
- checkpoint.pt: Estado completo
- samples_epoch_XXX.pt: Muestras generadas por epoca

---

## 11. Limitaciones y Mejoras

### Limitaciones
- Arquitectura MLP no explota la estructura espacial de pixeles
- Posible inestabilidad en el entrenamiento
- CPU puede ser lento

### Mejoras Propuestas
- DCGAN con capas convolucionales
- Normalizacion (BatchNorm)
- Tecnicas de estabilizacion

---

## 12. Conclusiones

- El Generador aprende una transformacion del espacio latente a imagenes
- El Discriminador aprende a distinguir imagenes reales de las generadas
- Las perdidas reflejan el equilibrio del juego adversario
- La evidencia de aprendizaje debe observarse en muestras visuales

---

## 13. Como Ejecutar

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

---

## 14. Referencias

- Goodfellow, I., et al. (2014). "Generative Adversarial Networks"
- MNIST Dataset: http://yann.lecun.com/exdb/mnist/

---

*Presentacion desarrollada con Python, PyTorch y Streamlit*