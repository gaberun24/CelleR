# CelleR - Evolutionary Single-Cell Organism Simulation

Real-time 2D evolution simulators built with Python + Pygame where single-celled organisms live, eat, reproduce, mutate, and evolve through natural selection.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Pygame](https://img.shields.io/badge/Pygame-2.0+-green) ![NumPy](https://img.shields.io/badge/NumPy-required-orange)

## CelleR 2 - NEAT Neural Network Evolution

**`cell_evolution2.py`** (~4700 lines) - Complete rewrite where **every decision is made by an evolving neural network**. No hardcoded behavior rules. Cells learn to eat, flee, hunt, and mate through NEAT (NeuroEvolution of Augmenting Topologies).

### Key differences from CelleR 1

| | CelleR 1 | CelleR 2 |
|---|---|---|
| **AI** | ~1500 lines of hardcoded behavior trees | NEAT neural networks (24 inputs, 18 outputs) |
| **Behavior** | Predictable, scripted | Emergent, evolved |
| **Genome** | 32-gene monolithic | Dual: BodyGenome (15 physical genes) + NeatGenome (topology + weights) |
| **Species** | Fixed herbivore/carnivore/omnivore | Continuous diet spectrum with NEAT-based speciation |
| **Training** | None | Pre-training arena with 4 scenarios |

### Dual Genome System

Each cell has two genomes:

**BodyGenome (hardware)** - 15 physical genes controlling:
- Size, cilia count/spread/power, metabolism, hue
- Diet (0=herbivore ... 1=carnivore), vision range, hearing range
- Horn size/count/spread, mouth size, turn rate

**NeatGenome (software)** - evolving neural network:
- Starts sparse (24 inputs directly connected to 18 outputs)
- NEAT mutations add hidden nodes and connections over time
- Topological sort (Kahn's algorithm) for valid feedforward evaluation
- 2 memory channels allow cells to maintain state between ticks
- Brain cost: larger networks drain more energy (evolution pressure for efficiency)

### Neural Network

**24 Inputs** (normalized): energy, HP, thirst, speed, nearest food/threat/prey/mate (inverted distance + angle), smell gradients (food, danger), nearby density, 2x memory

**18 Outputs** (tanh): 6x cilia power (positive-only), 6x cilia steering, attack/eat/mate/vocalize intents, 2x memory output

### Biological Systems

- **Directional mouth** - Gene-controlled cone angle (55-80 deg). Must face food to eat. Contact absorption if food overlaps body.
- **Continuous diet curve** - `plant_eff = (1-diet)^0.7`, `meat_eff = diet^0.7`. No magic thresholds.
- **Multi-horn system** - 0-3 horns with gene-controlled positions. Carnivore synergy (+30% attack), herbivore synergy (+50% defense).
- **Positive-only cilia** - Can only push, not pull. Visual animation reflects actual thrust direction.
- **Turn rate limit** - Genetic cap on angular velocity.
- **Pheromone system** - 7 types: trail, danger, food, mate, prey scent, corpse scent, ambush.

### Pre-Training Arena (F9)

The main simulation starts **empty**. Train cells in isolated arenas, save good genomes, then inject them into the world.

| Key | Scenario | What it trains | Cell type |
|---|---|---|---|
| 1 | Food Search | Directional eating, food navigation | Herbivores |
| 2 | Flee Training | Evading auto-predators | Small fast herbivores |
| 3 | Hunt Training | Killing prey (progressive difficulty) | Carnivores |
| 4 | Free Evolution | Mini-ecosystem survival | Mixed |

**Hunt Training** has progressive prey difficulty:
- 0-5k ticks: Stationary prey (learn to approach and attack)
- 5k-20k: Slow prey
- 20k-50k: Medium speed
- 50k+: Full speed

**Hall of Fame**: If all cells die, the best-ever genome respawns (1 clone + 4 mutated variants).

### Genome Library

Save trained genomes for injection into the main world:
- **S** / **Shift+S**: Save best / top 5 genomes (filtered by scenario: hunt=carnivores only, etc.)
- **C**: Clear library
- **Ctrl+I** (in main sim): Inject all saved genomes
- **Delete** (in main sim): Clear all cells

Files saved to `celler2_genomes/` with names like `genome_003_carn_hunt_training_f45.json`.

### Screens

| Key | Screen |
|---|---|
| F1 | Help overlay |
| F5 | Simulation view |
| F6 | Species list (NEAT speciation) |
| F7 | Brain / NEAT topology (hover nodes for connection details, Bezier curves) |
| F8 | Lineage tree |
| F9 | Pre-training arena |
| F10 | Settings menu |

### Controls

| Key | Action |
|---|---|
| Space | Pause / Resume |
| 1-4 | Speed: 1x / 2x / 5x / 10x |
| Click | Select cell |
| Tab | Action cam (follow selected cell) |
| Scroll | Zoom in / out |
| Arrows / WASD | Move camera |
| Delete | Clear all cells |
| Ctrl+I | Inject saved genomes |
| Ctrl+S / Ctrl+L | Save / Load game |
| ESC | Quit (autosaves) |

---

## CelleR 1 - Behavior Tree AI

**`cell_evolution.py`** (~7600 lines) - The original version with hardcoded AI behavior trees.

### Features
- **17-gene genetic system** controlling size, diet, cilia, combat stats, stealth, metabolism
- **Smart Predator AI** with 7 behavioral phases: rest, scavenge, stalk, ambush, chase, give up, patrol
- **Pack hunting** with alpha coordination via taunt system
- **Herbivore herd defense** with migration and danger memory
- **Sonar ping communication** (mate/attack/flee taunts as expanding ring waves)
- **Stealth system** for ambush predators
- **Hibernation** for predators/omnivores at low energy
- **Action camera** auto-zooming to combat and dramatic events

### Controls

| Key | Action |
|---|---|
| Space | Pause / Resume |
| W / S | Speed up / Slow down |
| Arrow Keys | Move camera |
| Scroll | Zoom in / out |
| Click | Select cell |
| C | Toggle action camera |
| F | Add food clusters |
| 1 / 2 / 3 | Spawn herbivores / predators / omnivores |
| H | Toggle headless mode |
| M | Settings menu |
| Ctrl+S / Ctrl+L | Save / Load |

---

## Installation

```bash
pip install pygame numpy
```

## Running

```bash
# CelleR 2 (NEAT neural networks)
python cell_evolution2.py

# CelleR 1 (behavior trees)
python cell_evolution.py
```

Requires Python 3.10+, Pygame 2.0+, NumPy.

## License

MIT

---

*Built with [Claude Code](https://claude.ai/code)*
