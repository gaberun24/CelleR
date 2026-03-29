# CelleR - Evolutionary Single-Cell Organism Simulation

A real-time 2D evolution simulator built with Python and Pygame where single-celled organisms live, eat, reproduce, mutate, and evolve through natural selection. Watch herbivores and predators emerge from a primordial soup as they develop unique traits across generations.

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Pygame](https://img.shields.io/badge/Pygame-2.0+-green) ![NumPy](https://img.shields.io/badge/NumPy-required-orange)

## Features

### Genetic System (15 Genes)
Each cell carries a genome that controls its physical traits and behavior:
- **Size** — body radius, affects energy cost and combat
- **Sense Range** — how far the cell can detect food, mates, and threats
- **Attack / Defense** — combat stats for predator-prey encounters
- **Metabolism** — energy efficiency
- **Diet** — herbivore (0) to predator (1) spectrum
- **Cilia Count & Spread** — number of cilia and their arrangement (torpedo vs. jellyfish locomotion)
- **Cilia Power** — thrust force per cilium
- **Turn Rate** — rotational speed
- **Aggression & Social** — behavioral tendencies
- **Reproduction Threshold & Litter Size** — when and how many offspring

### Movement Physics
Cells move using a cilia-based propulsion system with two distinct locomotion styles:
- **Torpedo mode** (spread = 0): all cilia point forward for fast straight-line movement, slow turning
- **Jellyfish mode** (spread = 1): cilia spread around the body, enabling lateral movement and rapid direction changes at the cost of some forward speed
- Energy-efficient **thrust system**: cells adjust their "gas pedal" (0-1) based on context — cruising slowly when safe, sprinting when chasing prey or fleeing danger

### Ecosystem

- **Food Oases** — food grows in clusters (oases) that regenerate over time, with variable sizes from small grass to large bushes
- **Herbivores** — seek food using enhanced sensing range (1.6x), migrate when starving
- **Predators** — hunt prey using visual tracking and scent pheromone trails, opportunistically prefer carcasses over live prey
- **Corpse System** — killed prey becomes a carcass that multiple predators can feed on over time
- **Shelters** — hiding spots on the map where cells can take cover

### Pheromone System (6 Types)
Cells leave chemical trails that others can detect and follow:
| Pheromone | Purpose |
|-----------|---------|
| Trail | General movement trace |
| Danger | Predator warning signal |
| Food Here | Marks food locations |
| Ambush | Predator lurking spot |
| Mate | Growing cloud for mate-seeking cells |
| Prey Scent | Herbivore scent trail for predator tracking |

### Reproduction
- **Partner required** — cells must find a compatible mate nearby to reproduce (no asexual reproduction)
- **Pheromone-guided** — mate-seeking cells emit a growing pheromone cloud to attract partners
- **Genetic crossover** — offspring inherit a mix of both parents' genes
- **Mutation** — random gene variations drive evolution
- **Overcrowding penalty** — reproduction is suppressed in densely populated areas

### AI Behaviors
- **Herbivore AI**: seek food → flee predators → find mates → migrate when starving
- **Predator AI**: check for carcasses → hunt prey → track scent trails → ambush near shelters
- **Anti-circular wandering**: cells break out of movement loops with random direction changes
- **Migration system**: starving cells travel increasing distances to find new food sources

## Installation

```bash
pip install pygame numpy
```

## Usage

```bash
python cell_evolution.py
```

## Controls

| Key | Action |
|-----|--------|
| `SPACE` | Pause / Resume |
| `W` / `S` | Speed up / Slow down |
| `Arrow Keys` | Move camera |
| `Scroll` | Zoom in / out |
| `Click` | Select cell (view details) |
| `F` | Add food at camera center |
| `1` | Spawn 5 herbivores |
| `2` | Spawn 3 predators |
| `M` | Settings menu (live parameter tuning) |
| `R` | Reset simulation |
| `TAB` | Toggle info panel |
| `Q` / `ESC` | Quit |

## Settings Menu

Press `M` to open the live settings panel where you can adjust:
- **Food Energy** — energy value of food
- **Eat Speed** — how fast cells consume food
- **Food Regrow** — oasis regeneration rate
- **Reproduction Cost** — energy cost of reproduction
- **Mutation Rate** — chance of gene mutation
- **Mutation Strength** — magnitude of mutations

## How It Works

1. **Initialization**: 60 cells with random genomes spawn in a world with 18 food oases
2. **Each tick**: cells move, sense their environment, make AI decisions, eat, and lose energy
3. **Natural selection**: cells that find food and mates survive; those that don't, die
4. **Evolution**: over generations, traits that aid survival become more common
5. **Emergent behavior**: watch as herbivores develop better food-sensing, predators become faster, and population dynamics create boom-bust cycles

## Requirements

- Python 3.8+
- Pygame 2.0+
- NumPy

## License

MIT

---

*Built with Claude Code (claude.ai/code)*
