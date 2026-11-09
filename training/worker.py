"""Gestor de trabajos de entrenamiento en hilo, sin llamadas a Streamlit."""
from dataclasses import asdict
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
import traceback
import uuid
import json
from data.dataset import get_mnist_loader
from training.trainer import train
from utils.config import seed_everything
from utils.device import get_device


class TrainingManager:
    """Posee hilo, cola, cancelación y estado sincronizado de una ejecución."""
    def __init__(self):
        self.events = Queue(maxsize=2)
        self.lock = Lock()
        self.stop_event = Event()
        self.thread = None
        self.run_id = None
        self.status = "inactivo"
        self.latest = None

    def start(self, config, output_root="outputs", initial_state=None):
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise RuntimeError("Ya existe un entrenamiento activo.")
            self.events = Queue(maxsize=2)
            self.stop_event = Event()
            self.run_id = uuid.uuid4().hex[:10]
            self.status = "preparando"
            self.latest = None
            run_dir = Path(output_root) / self.run_id
            self.thread = Thread(target=self._run, args=(config, run_dir, initial_state), daemon=True)
            self.thread.start()
            return self.run_id

    def _publish(self, message):
        with self.lock:
            self.latest = message
            self.status = message.get("status", self.status)
        try:
            self.events.put_nowait(message)
        except Exception:
            try:
                self.events.get_nowait()
                self.events.put_nowait(message)
            except Exception:
                pass

    def _run(self, config, run_dir, initial_state):
        try:
            seed_everything(config.seed)
            self._publish({"status": "preparando", "message": "Cargando MNIST"})
            loader, _subset = get_mnist_loader(config.category, config.batch_size,
                                                num_workers=config.num_workers)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "config.json").write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            result = train(config, loader, run_dir, get_device(), self._publish,
                           self.stop_event, initial_state)
            self._publish({"status": result["status"], "history": result["history"],
                           "run_dir": result["run_dir"]})
        except Exception as exc:
            self._publish({"status": "error", "error": str(exc),
                           "traceback": traceback.format_exc()})

    def request_stop(self):
        self.stop_event.set()
        with self.lock:
            self.status = "deteniendo"

    def drain(self):
        messages = []
        while True:
            try:
                messages.append(self.events.get_nowait())
            except Empty:
                return messages
