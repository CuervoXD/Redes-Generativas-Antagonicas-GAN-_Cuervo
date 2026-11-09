"""Explorador educativo Streamlit para una GAN MLP entrenada sobre MNIST."""
from pathlib import Path
import copy
import json
import pandas as pd
import streamlit as st
import torch
from data.dataset import get_mnist_loader
from training.trainer import create_models, load_checkpoint
from training.worker import TrainingManager
from utils.config import TrainConfig, seed_everything
from utils.device import get_device
from visualization.charts import loss_figure, probability_figure
from visualization.images import image_grid, generate_images
from visualization.parameters import layer_parameter_report, parameter_count, state_dict_difference

st.set_page_config(page_title="Explorador GAN", page_icon="🧠", layout="wide")


@st.cache_resource
def get_manager():
    """Comparte el trabajador local entre reruns de Streamlit."""
    return TrainingManager()


manager = get_manager()
ROOT = Path(__file__).parent
OUTPUTS = ROOT / "outputs"


def show_parameter_panel(title, model, initial_state=None):
    st.subheader(title)
    st.caption(f"Parámetros entrenables: **{parameter_count(model):,}**")
    if initial_state:
        differences = state_dict_difference(initial_state, model.state_dict())
        st.caption(f"Cambio absoluto medio frente al inicio (capas lineales): {sum(differences.values()) / max(1, len(differences)):.6g}")
    reports = layer_parameter_report(model)
    names = [item["name"] for item in reports]
    selected = st.selectbox(f"Selecciona una capa de {title}", names, key=f"layer_{title}")
    item = next(info for info in reports if info["name"] == selected)
    left, right = st.columns(2)
    left.metric("Entrada → salida", f"{item['input_features']} → {item['output_features']}")
    right.metric("Parámetros", f"{item['parameters']:,}")
    st.write(f"Tipo: `{item['type']}` · forma de pesos: `{item['weight_shape']}` · forma de sesgos: `{item['bias_shape']}`")
    st.json(item["weight_stats"])
    col_a, col_b = st.columns(2)
    col_a.code("Pesos (muestra): " + str(item["weight_sample"]))
    col_b.code("Sesgos (muestra): " + str(item["bias_sample"]))


@st.fragment(run_every=1)
def live_status():
    """Actualiza estado y progreso sin reiniciar ni duplicar el hilo de entrenamiento."""
    manager.drain()
    with manager.lock:
        status, snapshot, run_id = manager.status, copy.copy(manager.latest), manager.run_id
    if snapshot and snapshot.get("history"):
        snapshot["history"] = copy.deepcopy(snapshot["history"])
    st.write(f"**Estado:** {status}  ·  **Ejecución:** {run_id or '—'}")
    if snapshot and snapshot.get("epoch"):
        st.progress(min(snapshot["epoch"] / max(1, snapshot["total_epochs"]), 1.0),
                    text=f"Época {snapshot['epoch']} / {snapshot['total_epochs']}")
        metrics = snapshot.get("metrics", {})
        cols = st.columns(4)
        cols[0].metric("G loss", f"{metrics.get('generator_loss', float('nan')):.4f}")
        cols[1].metric("D loss", f"{metrics.get('discriminator_loss', float('nan')):.4f}")
        cols[2].metric("D(real)", f"{metrics.get('real_probability', float('nan')):.3f}")
        cols[3].metric("D(fake)", f"{metrics.get('fake_probability', float('nan')):.3f}")
        if snapshot.get("history"):
            st.pyplot(loss_figure(snapshot["history"]), clear_figure=True)
        if snapshot.get("samples") is not None:
            st.pyplot(image_grid(snapshot["samples"], title="Muestras de esta época"), clear_figure=True)
    elif snapshot and snapshot.get("status") == "error":
        st.error(snapshot.get("error", "Error desconocido"))
        with st.expander("Traza técnica"):
            st.code(snapshot.get("traceback", ""))
    elif status == "preparando":
        st.info("Preparando dataset / inicializando entrenamiento…")


st.title("Explorador Interactivo de Redes Generativas Antagónicas (GAN)")
st.caption("Aplicación educativa · MLP sobre una categoría de MNIST · Entrenamiento real en PyTorch")
page = st.sidebar.radio("Secciones", ["Inicio", "Arquitectura", "Configuración y entrenamiento",
                                      "Pérdidas", "Parámetros y pesos", "Resultados", "Conclusiones"])
st.sidebar.info(f"Dispositivo detectado: **{get_device()}**")

if page == "Inicio":
    st.header("Aprender compitiendo")
    st.markdown("Una GAN contiene dos redes: el **generador** transforma ruido aleatorio en imágenes; el **discriminador** estima si una imagen es real. Ambos se entrenan en una competencia adversaria.")
    st.markdown("""```text
Ruido z ──► Generador G ──► imagen artificial ──┐
                                                ├──► Discriminador D ──► logit real/falso
Imagen real MNIST ─────────────────────────────┘
                       D enseña a G mediante gradientes
```""")
    st.info("Esta aplicación entrena una GAN para imitar una categoría seleccionada (por defecto, el dígito 7). No es un clasificador de dígitos y no predice etiquetas para nuevas imágenes.")
    st.markdown("**Objetivo:** observar arquitectura, entrenamiento adversario, pérdidas, parámetros reales y la evolución de muestras generadas por el modelo.")

elif page == "Arquitectura":
    st.header("Arquitectura de las redes")
    st.subheader("Generador")
    st.code("z (100) → Linear 256 → LeakyReLU → Linear 512 → LeakyReLU → Linear 1024 → LeakyReLU → Linear 784 → Tanh → imagen 1×28×28")
    st.write("El vector latente es una muestra normal aleatoria de 100 dimensiones. Cada capa densa aprende combinaciones de rasgos; LeakyReLU mantiene gradientes también para activaciones negativas. Tanh ajusta la salida a [-1,1], coincidente con la normalización de MNIST.")
    st.subheader("Discriminador")
    st.code("imagen 1×28×28 → Flatten 784 → Linear 512 → LeakyReLU → Linear 256 → LeakyReLU → Linear 1 → logit")
    st.write("El logit se interpreta mediante sigmoid como probabilidad solo para mostrar métricas. BCEWithLogitsLoss combina sigmoid y entropía cruzada de forma numéricamente estable.")
    st.markdown("Durante el paso de D, las imágenes de G se separan con `detach()`. Durante el paso de G, D no actualiza pesos; sus gradientes atraviesan D para informar a G cómo engañarlo.")

elif page == "Configuración y entrenamiento":
    st.header("Configuración y entrenamiento")
    st.warning("La primera ejecución descargará MNIST. En CPU, empieza con pocas épocas para verificar el flujo; el entrenamiento completo puede tardar varios minutos.")
    with st.form("train_config"):
        a, b, c = st.columns(3)
        category = a.selectbox("Categoría MNIST", list(range(10)), index=7)
        epochs = b.number_input("Épocas", min_value=1, max_value=200, value=30)
        batch_size = c.select_slider("Batch size", options=[32, 64, 128, 256], value=128)
        d, e, f = st.columns(3)
        learning_rate = d.number_input("Learning rate", min_value=0.00001, max_value=0.01, value=0.0002, format="%.5f")
        latent_dim = e.number_input("Dimensión latente", min_value=8, max_value=512, value=100, step=8)
        seed = f.number_input("Semilla", min_value=0, max_value=2**31-1, value=42)
        submitted = st.form_submit_button("Iniciar nueva ejecución", type="primary", use_container_width=True)
    if submitted:
        try:
            config = TrainConfig(category=int(category), epochs=int(epochs), batch_size=int(batch_size),
                                 learning_rate=float(learning_rate), latent_dim=int(latent_dim), seed=int(seed))
            config.validate()
            seed_everything(config.seed)
            run_id = manager.start(config, str(OUTPUTS))
            st.session_state["active_run"] = run_id
            st.session_state["active_config"] = config.to_dict()
            st.success(f"Ejecución {run_id} iniciada para la categoría {category}.")
        except Exception as exc:
            st.error(str(exc))
    if manager.thread and manager.thread.is_alive() and st.button("Solicitar detención"):
        manager.request_stop()
        st.warning("Se detendrá al completar el lote actual.")
    live_status()
    st.caption("El entrenamiento corre en un hilo independiente; los controles de visualización siguen disponibles mientras se procesan épocas.")

elif page in ("Pérdidas", "Resultados", "Parámetros y pesos"):
    st.header(page)
    run_dirs = sorted([p for p in OUTPUTS.glob("*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True) if OUTPUTS.exists() else []
    if not run_dirs:
        st.info("Todavía no hay ejecuciones persistidas. Inicia un entrenamiento para crear métricas, pesos y muestras.")
    else:
        selected_dir = st.selectbox("Ejecución a consultar", run_dirs, format_func=lambda p: p.name)
        checkpoint_path = selected_dir / "checkpoint.pt"
        config_path = selected_dir / "config.json"
        if config_path.exists():
            run_config = json.loads(config_path.read_text(encoding="utf-8"))
        elif checkpoint_path.exists():
            try:
                run_config = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["config"]
            except Exception:
                run_config = {"category": "desconocida", "latent_dim": 100}
        else:
            run_config = {"category": "desconocida", "latent_dim": 100}
        st.caption(f"Categoría entrenada: **{run_config.get('category', '?')}** · Dimensión latente: {run_config.get('latent_dim', '?')}")
        csv_path = selected_dir / "metrics.csv"
        history = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
        if page == "Pérdidas":
            if history:
                latest = history[-1]
                cols = st.columns(4)
                cols[0].metric("Época", int(latest["epoch"]))
                cols[1].metric("Pérdida G", f"{latest['generator_loss']:.4f}")
                cols[2].metric("Pérdida D", f"{latest['discriminator_loss']:.4f}")
                cols[3].metric("Imágenes procesadas", int(latest["images_processed"]))
                st.pyplot(loss_figure(history), clear_figure=True)
                st.pyplot(probability_figure(history), clear_figure=True)
                st.dataframe(pd.DataFrame(history), use_container_width=True)
                st.markdown("Las pérdidas reflejan el equilibrio del juego, no una optimización convencional: no necesariamente disminuyen simultáneamente ni deben llegar a cero. Una D muy segura puede dificultar el aprendizaje de G; revisa también las imágenes y las probabilidades.")
            else:
                st.info("La ejecución aún no ha registrado una época completa.")
        elif page == "Resultados":
            st.subheader("Muestras generadas por época")
            sample_files = sorted(selected_dir.glob("samples_epoch_*.pt"))
            if sample_files:
                sample_file = st.selectbox("Época guardada", sample_files, format_func=lambda p: p.stem.replace("samples_epoch_", "Época "))
                st.pyplot(image_grid(torch.load(sample_file, map_location="cpu", weights_only=True), title=sample_file.stem), clear_figure=True)
            else:
                st.info("Aún no se han guardado muestras de una época.")
            st.subheader("Ejemplos reales del conjunto filtrado")
            try:
                loader, subset = get_mnist_loader(int(run_config["category"]), 16, num_workers=0)
                real_images, _ = next(iter(loader))
                st.pyplot(image_grid(real_images, title=f"MNIST real · clase {run_config['category']}"), clear_figure=True)
                st.caption(f"El subconjunto de entrenamiento contiene {len(subset):,} imágenes reales de la clase seleccionada.")
            except Exception as exc:
                st.warning(f"No se pudo cargar una vista del dataset: {exc}")
            if checkpoint_path.exists():
                if st.button("Generar 16 imágenes nuevas con ruido aleatorio"):
                    generator, discriminator = create_models(int(run_config["latent_dim"]), get_device())
                    load_checkpoint(checkpoint_path, generator, discriminator, map_location=get_device())
                    generated = generate_images(generator, int(run_config["latent_dim"]), 16, get_device())
                    st.pyplot(image_grid(generated, title="Nuevas muestras del generador entrenado"), clear_figure=True)
                    st.success("Se generaron 16 imágenes nuevas directamente con el checkpoint seleccionado.")
        else:
            st.markdown("Los **pesos** multiplican activaciones para aprender qué combinaciones de características son relevantes; los **sesgos** desplazan el umbral de activación. El optimizador los ajusta mediante gradientes de la función de pérdida.")
            if checkpoint_path.exists():
                generator, discriminator = create_models(int(run_config["latent_dim"]), get_device())
                load_checkpoint(checkpoint_path, generator, discriminator, map_location=get_device())
                initial_file = selected_dir / "initial_weights.pt"
                initial = torch.load(initial_file, map_location="cpu", weights_only=True) if initial_file.exists() else None
                left, right = st.columns(2)
                with left:
                    show_parameter_panel("Generador", generator, initial["generator"] if initial else None)
                with right:
                    show_parameter_panel("Discriminador", discriminator, initial["discriminator"] if initial else None)
                if initial is not None:
                    st.caption("Las estadísticas y muestras muestran valores reales de tensores del checkpoint; el cambio medio compara parámetros iniciales y guardados al final.")
            else:
                st.info("El checkpoint estará disponible al completar la primera época.")

elif page == "Conclusiones":
    st.header("Conclusiones y reflexión")
    st.markdown("- **Generador:** aprende una transformación del espacio latente a imágenes parecidas a la categoría de entrenamiento.\n- **Discriminador:** aprende a distinguir las imágenes reales del dataset de las creadas por G.\n- **Pérdidas:** describen objetivos opuestos; su evolución se interpreta junto con probabilidades y muestras visuales.\n- **Limitaciones:** MLP ignora explícitamente la vecindad espacial de píxeles; entrenar en CPU puede ser lento; GAN puede sufrir inestabilidad o colapso de modos.\n- **Mejoras:** usar DCGAN convolucional, normalización, técnicas de estabilización, evaluación cuantitativa y mayor variedad de datos.")
    st.info("La evidencia de aprendizaje debe observarse en muestras y métricas reales de cada ejecución; completar una época no garantiza por sí solo una GAN de buena calidad.")
