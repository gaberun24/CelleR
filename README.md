# CelleR - Evolutionary Single-Cell Organism Simulation

A real-time 2D evolution simulator built with Python and Pygame where single-celled organisms live, eat, reproduce, mutate, and evolve through natural selection. Watch herbivores, omnivores, and predators emerge from a primordial soup as they develop unique traits across generations.

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Pygame](https://img.shields.io/badge/Pygame-2.0+-green) ![NumPy](https://img.shields.io/badge/NumPy-required-orange)

## Features

### Genetic System (17 Genes)
Each cell carries a genome that controls its physical traits and behavior:
- **Size** — body radius, affects energy cost, HP, and combat
- **Sense Range** — how far the cell can detect food, mates, and threats
- **Attack / Defense** — combat stats for predator-prey encounters
- **Metabolism** — energy efficiency multiplier
- **Diet** — herbivore (0-0.3), omnivore (0.3-0.7), or predator (0.7-1.0) on a continuous spectrum
- **Cilia Count & Spread** — number of cilia and their arrangement (torpedo vs. jellyfish locomotion)
- **Cilia Power** — thrust force per cilium
- **Turn Rate** — rotational speed
- **Aggression & Social** — behavioral tendencies
- **Reproduction Threshold & Litter Size** — when and how many offspring
- **Taunt Power** — sonar-ping range and intensity for communication
- **Stealth** — how hard to detect when moving slowly (predator specialization)

### Movement Physics
Cells move using a cilia-based propulsion system with two distinct locomotion styles:
- **Torpedo mode** (spread = 0): all cilia point forward for fast straight-line movement, slow turning
- **Jellyfish mode** (spread = 1): cilia spread around the body, enabling lateral movement and rapid direction changes at the cost of some forward speed
- Energy-efficient **thrust system**: cells adjust their "gas pedal" (0-1) based on context

### Three Cell Types

- **Herbivores** (diet 0-0.3) — eat plants, enhanced food sensing (1.6x), flee from predators, herd defense
- **Omnivores** (diet 0.3-0.7) — eat plants and old carcasses, don't hunt live prey
- **Predators** (diet 0.7-1.0) — hunt herbivores and omnivores, pack hunting, ambush tactics

### Ecosystem

- **Food Oases** — food grows in clusters that regenerate, with burst regrowth when depleted
- **Corpse System** — dead cells become carcasses with freshness decay (predators eat fresh, omnivores eat old)
- **Shelters** — hiding spots for prey, ambush positions for predators
- **Obstacles** — rocks that block line-of-sight but not sound/taunts (Liang-Barsky intersection)
- **Reproductive Isolation** — only cells with diet difference < 0.15 can mate (prevents species blending)

### Taunt System (Sonar Ping Communication)
Cells emit expanding ring waves that trigger once as the wavefront passes:
- **Mate taunt** (pink) — attracts compatible mates from afar (predators emit 2x range)
- **Attack taunt** (red) — alpha predator orders pack to target specific prey, overrides existing targets
- **Flee taunt** (yellow) — panic signal causing allies to flee with zigzag evasion
- Hungry predators **eavesdrop** on prey taunts and are attracted to the sound

### Stealth System
- Predators with high stealth gene are harder to detect when moving slowly
- Detection formula considers: speed ratio, stealth gene, hiding bonus, distance
- Close range negates stealth — prey always spots predators nearby
- Trade-off: stealth costs extra energy to maintain

### Smart Predator AI
- **7 behavioral phases**: rest → scavenge → stalk → ambush → chase → give up → patrol
- **ROI hunting** — abandons chase if energy spent exceeds 35% of expected reward
- **Interceptor targeting** — aims where prey will be, not where it is (smoothed prediction)
- **Pack hunting** — alpha coordinates attacks via taunt, flanking at range
- **Target stickiness** — score bonus for current target prevents oscillation
- **Memory** — learns which prey types (weak, isolated, slow) are easiest to catch
- **Bite shock** — strong bites stun prey (5-25 ticks paralysis), weaker bites slow them (30-70% speed)

### Herbivore AI
- **Stealth-aware detection** — slow predators at distance go unnoticed
- **Tiered flee response** — panic sprint (very close), shelter seeking (close), alert retreat (far)
- **Herd defense** — large groups are braver; tank cells provide defense aura
- **Migration** — relocates when area becomes too dangerous (3+ attacks)
- **Memory** — remembers food spots and danger zones, shares with herd

### Pheromone System (7 Types)
| Pheromone | Purpose |
|-----------|---------|
| Trail | General movement trace |
| Danger | Predator warning signal |
| Food Here | Marks food locations |
| Ambush | Predator lurking spot |
| Mate | Growing cloud for mate-seeking cells |
| Prey Scent | Herbivore/omnivore trail for predator tracking |
| Corpse Scent | Expanding stench cloud from decaying carcasses |

### Well-Fed Sluggishness
- Herbivores and omnivores become slower and less alert when full (up to 35% debuff)
- `alertness = 1.0 - fullness * 0.35` affects both speed and sense range
- Creates natural boom-bust cycles: abundant food → fat prey → easy hunting → more predators
- Predators are always sharp — no sluggishness penalty

### Bounding Circle Collision
- Same-type cells can't overlap or pass through each other
- Mass-proportional push resolution (heavier cells move less)
- Predator-prey pairs excluded from collision (so attacks aren't blocked)
- Prevents unrealistic 50-cells-per-pixel clustering

### Action Camera
- Press `C` to toggle automatic cinematic camera
- Auto-zooms to exciting events: combat, pack hunts, chases, panic signals
- Event scoring system prioritizes the most dramatic moments
- Holds 5-10 seconds per event with smooth lerp transitions
- **Minimap** (180x180px, bottom-left) shows world overview when active

### Population Cap
- Configurable maximum cell count (default: 400, adjustable 50-1000 in settings)
- When cap is reached, the most populous species stops reproducing
- Other species can still reproduce, maintaining diversity
- Hard cap prevents runaway population even if blocking isn't enough

### Hibernation
- Predators/omnivores enter hibernation instead of dying (once per lifetime, max 500 ticks)
- Metabolism drops to 1%, awakens with adrenaline burst when prey enters range (costs 15% max HP)
- Omnivore hibernators also wake up for nearby food

### Fitness-Weighted Reproduction
- Both parents contribute energy to offspring (no energy black hole)
- Better parent's genes dominate (55-75% inheritance ratio)
- K-strategy (predators: 1 strong offspring) vs r-strategy (herbivores: 2-4 fast offspring)
- Memory inheritance: hunting grounds, food spots, danger zones pass to children

## Installation

```bash
pip install pygame numpy
```

## Usage

```bash
# English version
python cell_evolution_en.py

# Hungarian version (eredeti)
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
| `C` | Toggle action camera (auto-zoom to events + minimap) |
| `F` | Add food clusters |
| `1` / `2` / `3` | Spawn herbivores / predators / omnivores |
| `H` | Toggle headless mode (max simulation speed, no rendering) |
| `M` | Settings menu (live parameter tuning) |
| `TAB` | Toggle stats panel |
| `Ctrl+S` | Save world state to JSON |
| `Ctrl+L` | Load world state from JSON |
| `R` | Reset simulation |
| `Q` / `ESC` | Quit |

### Headless Mode
Press `H` to run the simulation at maximum speed without rendering. The display updates periodically to show progress. Use `+`/`-` to adjust render interval.

## Settings Menu

Press `M` to open the live settings panel:
- **Food Energy** — energy value of food items
- **Eating Speed** — how fast cells consume food per tick
- **Food Regrowth** — oasis regeneration rate
- **Reproduction Cost** — energy fraction spent on reproduction
- **Mutation Rate** — chance of gene mutation per gene
- **Mutation Strength** — magnitude of mutations
- **Max Population** — population cap (50-1000)

## Architecture

Single-file Python application (~4200 lines):

| Class | Purpose |
|-------|---------|
| `Genome` | 17-gene genetic code with properties, crossover, mutation |
| `Cell` | Organism: physics, HP, sprint, memory, reproduction |
| `PheromoneMap` | Grid-based chemical signaling (7 types, gradient reading) |
| `SpatialGrid` | Spatial hashing for O(1) neighbor queries |
| `World` | Simulation engine: AI, food, combat, spawning, save/load |
| `Renderer` | Pygame visualization: camera, HUD, graphs, pheromone overlay |
| `SettingsMenu` | Live-adjustable simulation parameters with sliders |
| `Game` | Main loop, input handling, headless mode |

## Requirements

- Python 3.8+
- Pygame 2.0+
- NumPy

## License

MIT

---

*Built with [Claude Code](https://claude.ai/code)*
