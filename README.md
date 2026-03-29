# CelleR - Evolutionary Single-Cell Organism Simulation

A real-time 2D evolution simulator built with Python and Pygame where single-celled organisms live, eat, reproduce, mutate, and evolve through natural selection. Watch herbivores, omnivores, and predators emerge from a primordial soup as they develop unique traits across generations.

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Pygame](https://img.shields.io/badge/Pygame-2.0+-green) ![NumPy](https://img.shields.io/badge/NumPy-required-orange)

## Features

### Genetic System (15 Genes)
Each cell carries a genome that controls its physical traits and behavior:
- **Size** — body radius, affects energy cost and combat
- **Sense Range** — how far the cell can detect food, mates, and threats
- **Attack / Defense** — combat stats for predator-prey encounters
- **Metabolism** — energy efficiency
- **Diet** — herbivore (0-0.3), omnivore (0.3-0.7), or predator (0.7-1.0) on a continuous spectrum
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

### Three Cell Types

- **Herbivores** (diet 0-0.3) — eat plants, enhanced food sensing range (1.6x), flee from predators, herd defense mechanics
- **Omnivores** (diet 0.3-0.7) — eat both plants and carcasses but don't hunt live prey, versatile survival strategy
- **Predators** (diet 0.7-1.0) — hunt herbivores and omnivores, never attack other predators, opportunistically prefer carcasses

### Ecosystem

- **Food Oases** — food grows in clusters (oases) that regenerate over time, with variable sizes from small grass to large bushes
- **Corpse System** — killed prey becomes a carcass that multiple predators/omnivores can feed on over time
- **Shelters** — hiding spots on the map where cells can take cover or set ambushes

### Pheromone System (6 Types)
Cells leave chemical trails that others can detect and follow:
| Pheromone | Purpose |
|-----------|---------|
| Trail | General movement trace |
| Danger | Predator warning signal |
| Food Here | Marks food locations |
| Ambush | Predator lurking spot |
| Mate | Growing cloud for mate-seeking cells |
| Prey Scent | Herbivore/omnivore scent trail for predator tracking |

### Smart Predator AI
- **Interceptor targeting** — predators aim where the prey *will be*, not where it currently is, based on velocity prediction
- **Speed assessment** — won't chase prey that's faster unless sprint is available
- **ROI hunting** — tracks energy spent on a chase; gives up if cost exceeds 35% of expected carcass value
- **Proactive oasis patrol** — when hungry with no prey in sight, moves toward oases where herbivores gather
- **Pack hunting** — social predators share targets and flank prey from multiple angles
- **Ambush tactics** — lurk in shelters near oases and burst-sprint at approaching prey

### Hibernation / Spore State
- Predators and omnivores that reach critical energy enter **hibernation** instead of dying immediately
- Metabolism drops to 1%, all movement stops
- Awakens with an adrenaline burst (sprint) when prey enters sensing range
- **Once per lifetime** — can only hibernate once, max 500 ticks, creating genuine last-chance survival drama

### Reproduction
- **Partner required** — cells must find a compatible mate nearby to reproduce (no asexual reproduction)
- **Pheromone-guided** — mate-seeking cells emit a growing pheromone cloud to attract partners
- **Genetic crossover** — offspring inherit a mix of both parents' genes
- **Mutation** — random gene variations drive evolution
- **Overcrowding penalty** — reproduction is suppressed in densely populated areas

### Visual Feedback
- **Selected cell pheromone overlay** — click any cell to see its surrounding pheromone landscape with color-coded legend
- **Visible oases** — green glow with intensity based on richness
- **Shelter markers** — decorated circles with branch patterns
- **Hibernation indicator** — pulsing grey ring with "z" marker

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
| `Click` | Select cell (view details + pheromone overlay) |
| `F` | Add food at camera center |
| `1` | Spawn 5 herbivores |
| `2` | Spawn 3 predators |
| `3` | Spawn 4 omnivores |
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
5. **Emergent behavior**: watch as herbivores develop better food-sensing, predators become faster hunters, omnivores find their niche, and population dynamics create boom-bust cycles

## Requirements

- Python 3.8+
- Pygame 2.0+
- NumPy

## License

MIT

---

*Built with Claude Code (claude.ai/code)*
