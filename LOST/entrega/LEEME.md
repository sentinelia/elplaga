# Obligatorio Marzo 2026 — Inteligencia Artificial — Proyecto LOST

## Contenido de la entrega

| Ruta | Contenido |
|---|---|
| `Informe_Proyecto_LOST.pdf` | Informe (9 páginas): enfoque, experimentos, resultados, conclusiones, notas de advertencia y declaración de uso de IA |
| `MountainCarContinuous/` | Código Python listo para ejecutar (entorno Poetry del curso) |
| `MountainCarContinuous/continuous_mountain_car.ipynb` | Notebook ejecutada que recorre las 4 tareas con tablas, figuras y análisis |
| `MountainCarContinuous/models/*.pkl` | **Modelos computados** (obligatorio): `final_qlearning.pkl` es el principal (retorno 93.4 ± 0.2, 100% de éxito) |
| `MountainCarContinuous/results/` | Resúmenes por corrida (`*_summary.csv`), historias por episodio (`history_*.csv`) y figuras (`results/es/` en español) |

## Cómo ejecutar

```bash
cd MountainCarContinuous
poetry install            # o: pip install numpy gymnasium matplotlib
poetry run jupyter notebook continuous_mountain_car.ipynb
```

Reproducción de los experimentos (todos sembrados):

```bash
python experiments.py discretization      # Tarea 1 (5 acciones)
python experiments.py discretization_a4   # Tarea 1 (grillas sobre 4 acciones)
python experiments.py hyperparams         # Tarea 3 (5 acciones)
python experiments.py hyperparams_a4      # Tarea 3 (discretización elegida)
python experiments.py dynaq               # Tarea 4 (Dyna-Q vs Q-Learning)
python experiments.py final               # Modelos finales -> models/*.pkl
python make_plots.py --lang es            # Regenera las figuras en español
```

## Uso de un modelo entrenado

```python
import gymnasium as gym
from q_learning_agent import QLearningAgent

agent = QLearningAgent.load("models/final_qlearning.pkl")
env = gym.make("MountainCarContinuous-v0", render_mode="human")
print(agent.test_agent(env, episodes=5))
```

## Declaración de uso de IA generativa

Se utilizó Claude (Anthropic) como asistente de programación y redacción:
implementación de los agentes y del arnés de experimentos según las técnicas
de la letra (Sutton & Barto §6.5 y §8.1–8.2), ejecución de los experimentos,
generación de figuras y redacción inicial del informe. Todo el contenido fue
revisado y verificado contra los datos crudos de `results/`. Detalle completo
en la sección 10 del informe.
