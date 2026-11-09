# Explorador Interactivo de Redes Generativas Antagónicas (GAN)

Aplicación educativa en español que implementa y entrena una GAN MLP real con PyTorch sobre una categoría seleccionable de MNIST. La interfaz Streamlit presenta arquitectura, pérdidas, métricas, muestras, parámetros y resultados persistidos.

## Objetivo general

Comprender experimentalmente cómo un generador aprende a producir imágenes similares a una categoría real y cómo un discriminador aprende a distinguir imágenes reales de artificiales.

## Objetivos específicos

- Implementar Generator y Discriminator como redes neuronales independientes.
- Filtrar MNIST para entrenar exclusivamente una categoría de 0 a 9.
- Registrar pérdidas, probabilidades, tiempo e imágenes procesadas.
- Inspeccionar pesos y sesgos obtenidos de tensores reales antes y después del entrenamiento.
- Persistir checkpoints, métricas CSV y muestras para comparar épocas.

## Fundamentos teóricos

Una GAN enfrenta dos modelos. El generador \(G(z)\) recibe un vector latente aleatorio \(z\) y crea una imagen. El discriminador \(D(x)\) asigna evidencia de autenticidad a una imagen. Durante el entrenamiento, D mejora al separar reales de falsas, mientras G mejora al engañar a D. El entrenamiento es adversario y las pérdidas pueden oscilar; no se espera necesariamente que ambas disminuyan simultáneamente o lleguen a cero.

### Fórmulas

\[
\mathcal{L}_D = -\mathbb{E}_{x\sim p_{data}}[\log D(x)] - \mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]
\]

\[
\mathcal{L}_G = -\mathbb{E}_{z\sim p_z}[\log D(G(z))]
\]

En código se utiliza `nn.BCEWithLogitsLoss`, que combina sigmoid y Binary Cross Entropy de manera numéricamente estable. D devuelve logits; sigmoid se aplica al mostrar probabilidades.

## Arquitectura

**Generador:** vector latente (100 por defecto) → Linear 256 → LeakyReLU → Linear 512 → LeakyReLU → Linear 1024 → LeakyReLU → Linear 784 → Tanh → imagen 1×28×28.

**Discriminador:** imagen 1×28×28 → aplanado 784 → Linear 512 → LeakyReLU → Linear 256 → LeakyReLU → Linear 1 (logit real/falso).

Las imágenes se normalizan de [0,1] a [-1,1], rango coincidente con Tanh. Para entrenar D, las imágenes falsas se separan del grafo del generador con `detach()`. Al entrenar G se congelan los parámetros de D, pero se conserva el gradiente a través de D hasta G.

## Dataset

Se utiliza MNIST de torchvision. La descarga sucede automáticamente si faltan los archivos locales. El filtro retiene exclusivamente el dígito elegido en la interfaz (predeterminado: **7**). Esto es una GAN de una categoría, no un clasificador de dígitos.

## Tecnologías

Python 3.10+, PyTorch, torchvision, Streamlit, NumPy, Pandas y Matplotlib. La aplicación detecta CUDA automáticamente y utiliza CPU en su ausencia. En Windows el `DataLoader` utiliza `num_workers=0` por compatibilidad.

## Parámetros predeterminados

- Latent dimension: 100
- Batch size: 128
- Learning rate: 0.0002
- Adam betas: (0.5, 0.999)
- Épocas: 30
- Resolución: 28×28, escala de grises
- Semilla: 42

Épocas, batch size, tasa de aprendizaje, dimensión latente, categoría y semilla se modifican desde la interfaz. El tiempo en CPU depende del equipo.

## Estructura

```text
GAN/
├── app.py
├── requirements.txt
├── README.md
├── models/              # Generator y Discriminator
├── training/            # BCE, métricas, ciclo de entrenamiento y worker
├── data/                # Dataset MNIST filtrado
├── visualization/       # Gráficas, imágenes e inspección de tensores
├── utils/               # Configuración y detección de dispositivo
├── tests/               # Pruebas de formas, pérdidas, entrenamiento y checkpoint
├── checkpoints/         # Reservado para artefactos
└── outputs/             # Ejecuciones con CSV, pesos y muestras
```

## Instalación y ejecución

En Windows PowerShell o Linux/macOS:

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Para CUDA, instalar primero el paquete PyTorch/torchvision compatible con el controlador desde [pytorch.org](https://pytorch.org/get-started/locally/) y luego instalar el resto de dependencias. La aplicación siempre dispone de fallback CPU.

## Persistencia y reproducibilidad

Cada ejecución recibe un ID único bajo `outputs/<id>/` y conserva `config.json`, `metrics.csv`, `initial_weights.pt`, `checkpoint.pt` y muestras `samples_epoch_NNN.pt`. El checkpoint guarda estados de ambas redes y optimizadores, época, historial, ruido fijo, configuración y estados RNG para reanudar entrenamiento mediante `training.trainer.train(..., initial_state=...)`. Una reanudación debe usar una configuración compatible, especialmente la dimensión latente. Las semillas Python, NumPy y PyTorch se fijan antes de la ejecución.

## Resultados esperados

Tras una época se encuentran pérdidas y muestras del generador; la calidad inicial puede parecer ruido y mejora de manera no monótona. La aplicación muestra ejemplos reales del subconjunto elegido, samples de distintas épocas y nuevas imágenes producidas con el checkpoint. La calidad se evalúa visualmente junto con las métricas; finalizar el ciclo no prueba por sí solo que el modelo haya convergido.

## Pruebas

```bash
python -m pytest -q
```

Las pruebas son sintéticas y no requieren descargar MNIST. Comprueban dimensiones, diferenciabilidad de pérdidas, cambio de pesos durante un entrenamiento corto y round-trip de checkpoint. La prueba manual de integración consiste en iniciar desde Streamlit una ejecución breve, verificar los artefactos en outputs y consultar las distintas secciones.

## Limitaciones y mejoras

La arquitectura MLP aplana píxeles y no representa explícitamente su vecindad espacial; puede producir dígitos borrosos o repetitivos, y el entrenamiento GAN es inestable. Una extensión natural es DCGAN convolucional, técnicas de estabilización, evaluación cuantitativa (por ejemplo FID con cautela para MNIST) y soporte de conjuntos mayores. El gestor de trabajo es local al proceso Streamlit; despliegues multiproceso requieren una cola externa o estado compartido.

## Conclusiones

El proyecto hace visible el aprendizaje competitivo mediante imágenes, pérdidas y cambios en parámetros reales. El generador modela patrones de una categoría concreta; el discriminador aprende evidencia para diferenciar datos reales y sintéticos. Las muestras y su evolución aportan evidencia complementaria a métricas que no tienen una interpretación de convergencia tan simple como en aprendizaje supervisado.
