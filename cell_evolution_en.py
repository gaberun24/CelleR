#!/usr/bin/env python3
"""
Cell Evolution Simulation - Primordial Soup Ecosystem

Single-celled organisms live in a 2D world: they feed, move, reproduce,
and mutate. Predators and herbivores evolve through natural selection.

Controls:
  SPACE       - Pause/Resume
  W/S         - Speed +/-
  Arrows      - Camera movement
  F           - Add food
  1           - Herbivore spawn (5 at camera center)
  2           - Predator spawn (3 at camera center)
  3           - Omnivore spawn (4 at camera center)
  M           - Settings menu
  R           - Reset
  TAB         - Info panel on/off
  Click       - Select | Scroll: Zoom
  Ctrl+S      - Save (JSON)
  Ctrl+L      - Load (JSON)
  Q/ESC       - Quit
"""

import pygame
import numpy as np
import random
import math
import sys
import json
import os
from collections import defaultdict

# --- Constants ---
FPS = 60
FOOD_COLOR = (50, 200, 50)
FOOD_RADIUS = 3
FOOD_ENERGY = 28
MAX_FOOD = 500
FOOD_SPAWN_RATE = 0  # No random spawn! Only appears at oasis clusters
INITIAL_CELLS = 60
INITIAL_FOOD = 18  # Number of starting oases
MIN_CELL_ENERGY = 0
REPRODUCTION_COST_RATIO = 0.45  # Energy ratio lost during reproduction
MUTATION_RATE = 0.15
MUTATION_STRENGTH = 0.12
CORPSE_ENERGY_RATIO = 0.7  # More energy from corpses (better for predators)
OVERCROWDING_RADIUS = 50    # Cells within this radius count as crowding
OVERCROWDING_THRESHOLD = 5  # Penalty above this many cells
OVERCROWDING_PENALTY = 0.03 # Extra energy drain per neighbor
SPATIAL_GRID_SIZE = 60  # Spatial grid size for collision detection

# Pheromone system
PHEROMONE_DECAY = 0.995     # Pheromone decay per tick
PHEROMONE_MAX_AGE = 600     # Max pheromone lifetime
PHEROMONE_GRID_SIZE = 40    # Pheromone grid size

# Shelter / cover
NUM_SHELTERS = 10          # Number of shelters on the map
SHELTER_RADIUS = 35        # Shelter radius
AMBUSH_ENERGY_MULT = 0.1   # Ambushing predator energy multiplier (90% savings)

# Obstacles (rocks — block line of sight, but not sound/taunt)
NUM_OBSTACLES = 8          # Number of rocks
OBSTACLE_MIN_SIZE = 30     # Minimum size
OBSTACLE_MAX_SIZE = 80     # Maximum size

# Eating system
EAT_SPEED = 1.0            # Energy extracted per tick
FOOD_MIN_ENERGY = 0.5      # Food disappears below this

# Color palette for species
def hsl_to_rgb(h, s, l):
    """HSL -> RGB conversion."""
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = l - c / 2
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))


# --- Genome ---
class Genome:
    """Genetic code of a single-celled organism."""

    GENE_NAMES = [
        "size",          # 0: Size (3-20)
        "sense_range",   # 1: Sense range (20-200)
        "attack",        # 2: Attack power (0-10)
        "defense",       # 3: Defense (0-10)
        "metabolism",    # 4: Metabolism efficiency (0.3-1.0)
        "diet",          # 5: Diet (0=herbivore, 1=predator)
        "repro_thresh",  # 6: Reproduction energy threshold (50-200)
        "hue",           # 7: Color hue (0-360)
        "aggression",    # 8: Aggression (0-1)
        "social",        # 9: Social (0-1)
        "num_cilia",     # 10: Number of cilia (1-8)
        "cilia_spread",  # 11: Cilia spread (0=one direction, 1=radial)
        "cilia_power",   # 12: Power per cilium (0.3-2.0)
        "turn_rate",     # 13: Turn rate (0.02-0.2 rad/tick)
        "litter_size",   # 14: Litter size (1-4 offspring at once)
        "taunt_power",   # 15: Taunt power (0.1-2.0) — wave range and intensity
        "stealth",       # 16: Stealth (0-1) — harder to detect when moving slowly
    ]

    def __init__(self, genes=None):
        if genes is not None:
            self.genes = np.array(genes, dtype=float)
        else:
            self.genes = np.array([
                random.uniform(5, 12),      # size
                random.uniform(120, 250),   # sense_range (detect from farther)
                random.uniform(0.5, 4),     # attack
                random.uniform(0.5, 3),     # defense
                random.uniform(0.4, 0.8),   # metabolism
                random.uniform(0, 1),       # diet
                random.uniform(40, 90),     # repro_thresh
                random.uniform(0, 360),     # hue
                random.uniform(0.2, 0.8),   # aggression
                random.uniform(0.1, 0.5),   # social
                random.uniform(1, 6),       # num_cilia
                random.uniform(0, 1),       # cilia_spread
                random.uniform(0.5, 1.5),   # cilia_power
                random.uniform(0.03, 0.12), # turn_rate
                random.uniform(1, 2),       # litter_size
                random.uniform(0.2, 1.2),   # taunt_power (volume)
                random.uniform(0.0, 0.5),   # stealth (stealth ability)
            ], dtype=float)

    @property
    def size(self): return np.clip(self.genes[0], 3, 25)
    @property
    def sense_range(self): return np.clip(self.genes[1], 15, 350)
    @property
    def attack(self): return np.clip(self.genes[2], 0, 15)
    @property
    def defense(self): return np.clip(self.genes[3], 0, 15)
    @property
    def metabolism(self): return np.clip(self.genes[4], 0.2, 1.0)
    @property
    def diet(self): return np.clip(self.genes[5], 0, 1)
    @property
    def repro_thresh(self): return np.clip(self.genes[6], 40, 300)
    @property
    def hue(self): return self.genes[7] % 360
    @property
    def aggression(self): return np.clip(self.genes[8], 0, 1)
    @property
    def social(self): return np.clip(self.genes[9], 0, 1)
    @property
    def num_cilia(self): return int(np.clip(round(self.genes[10]), 1, 8))
    @property
    def cilia_spread(self): return np.clip(self.genes[11], 0, 1)
    @property
    def cilia_power(self): return np.clip(self.genes[12], 0.2, 3.0)
    @property
    def turn_rate(self): return np.clip(self.genes[13], 0.01, 0.25)
    @property
    def litter_size(self): return int(np.clip(round(self.genes[14]), 1, 4))
    @property
    def taunt_power(self): return np.clip(self.genes[15] if len(self.genes) > 15 else 0.5, 0.05, 2.5)
    @property
    def stealth(self): return np.clip(self.genes[16] if len(self.genes) > 16 else 0.2, 0.0, 1.0)

    @property
    def max_speed(self):
        """Max speed calculated from cilia configuration.
        Torpedo (spread=0): fast in a straight line, slow turning
        Jellyfish (spread=1): slower straight, but can go any direction instantly
        """
        # Spread effect: torpedo is more efficient in straight line
        # BUT jellyfish doesn't get a big penalty — only ~25% slower forward
        forward_efficiency = 1.0 - self.cilia_spread * 0.25
        base_speed = self.num_cilia * self.cilia_power * forward_efficiency * 0.4
        # Predators are more aerodynamic
        predator_boost = 1.0 + self.diet * 0.35
        return base_speed * predator_boost

    @property
    def maneuverability(self):
        """How well it can move sideways (based on spread).
        0 = forward only, 1 = full power in any direction.
        """
        return self.cilia_spread

    def cilia_positions(self):
        """Cilia positions around the cell (angle list), 0 = forward."""
        n = self.num_cilia
        spread = self.cilia_spread
        if n == 1:
            return [0.0]
        # spread=0: all at 0 degrees (forward), spread=1: evenly around
        angles = []
        total_arc = spread * 2 * math.pi  # from 0 to 2*pi
        if total_arc < 0.01:
            return [0.0] * n
        start = -total_arc / 2
        for i in range(n):
            angle = start + (total_arc * i / (n - 1)) if n > 1 else 0
            angles.append(angle)
        return angles

    def is_predator(self):
        return self.diet > 0.7

    def is_omnivore(self):
        return 0.3 < self.diet <= 0.7

    def is_herbivore(self):
        return self.diet <= 0.3

    def color(self):
        sat = 0.7 + 0.3 * self.diet
        light = 0.35 + 0.15 * (1 - self.diet)
        return hsl_to_rgb(self.hue, sat, light)

    def mutate(self):
        new_genes = self.genes.copy()
        for i in range(len(new_genes)):
            if random.random() < MUTATION_RATE:
                scale = abs(new_genes[i]) * MUTATION_STRENGTH + 0.1
                new_genes[i] += random.gauss(0, scale)
        return Genome(new_genes)

    def crossover(self, other, self_fitness=0.5, other_fitness=0.5):
        """Fitness-weighted crossover: better parent's genes are inherited more often.
        For some genes, takes the weighted average of both parents (blend)."""
        total = self_fitness + other_fitness + 0.001
        # Better parent's genes are inherited 55-75% of the time
        self_ratio = np.clip(self_fitness / total, 0.3, 0.7)

        child_genes = np.empty_like(self.genes)
        for i in range(len(self.genes)):
            r = random.random()
            if r < 0.25:
                # Blend: weighted average (smoother inheritance, especially for size/speed type genes)
                child_genes[i] = self.genes[i] * self_ratio + other.genes[i] * (1.0 - self_ratio)
            elif r < 0.25 + self_ratio * 0.75:
                # Dominant parent's gene
                child_genes[i] = self.genes[i]
            else:
                # Recessive parent's gene
                child_genes[i] = other.genes[i]
        return Genome(child_genes)


# --- Pheromone system ---
class PheromoneMap:
    """Chemical signals on the map — cells leave pheromones."""
    TRAIL = 0       # Trail tracking (I was here)
    DANGER = 1      # Danger (predator was here)
    FOOD_HERE = 2   # Food is/was here
    AMBUSH = 3      # Ambush (predator is watching)
    MATE = 4        # Mating pheromone (I want to reproduce!)
    PREY_SCENT = 5  # Prey scent (herbivore trail, detected by predators)
    CORPSE_SCENT = 6  # Corpse scent (carcass stench, growing cloud)

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid_size = PHEROMONE_GRID_SIZE
        self.cols = width // self.grid_size + 1
        self.rows = height // self.grid_size + 1
        # Each cell: {type: intensity} dict
        self.grid = [[{} for _ in range(self.cols)] for _ in range(self.rows)]

    def deposit(self, x, y, ptype, intensity=1.0):
        """Deposit pheromone at a point."""
        col = int(x / self.grid_size)
        row = int(y / self.grid_size)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            cell = self.grid[row][col]
            cell[ptype] = min(cell.get(ptype, 0) + intensity, 5.0)

    def deposit_cloud(self, x, y, ptype, intensity=1.0, radius=3):
        """Pheromone cloud — spreads widely, stronger at center."""
        col = int(x / self.grid_size)
        row = int(y / self.grid_size)
        radius_sq = radius * radius
        inv_radius = 1.0 / (radius + 1)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = row + dr, col + dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    dist_sq = dr * dr + dc * dc
                    if dist_sq <= radius_sq:
                        # Fast approximation instead of sqrt: Manhattan-ish falloff
                        falloff = max(0.1, 1.0 - (abs(dr) + abs(dc)) * 0.5 * inv_radius)
                        cell = self.grid[r][c]
                        cell[ptype] = min(cell.get(ptype, 0) + intensity * falloff, 5.0)

    def read(self, x, y, ptype, radius=1):
        """Read pheromone intensity at a point (including neighboring cells)."""
        col = int(x / self.grid_size)
        row = int(y / self.grid_size)
        total = 0
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = row + dr, col + dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    total += self.grid[r][c].get(ptype, 0)
        return total

    def read_gradient(self, x, y, ptype):
        """Pheromone gradient direction (where it gets stronger)."""
        col = int(x / self.grid_size)
        row = int(y / self.grid_size)
        gx, gy = 0.0, 0.0
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                r, c = row + dr, col + dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    val = self.grid[r][c].get(ptype, 0)
                    if val > 0:
                        gx += dc * val
                        gy += dr * val
        return gx, gy

    # Decay compensation: called every 3rd tick, so 3x exponentiation
    _DECAY_COMPENSATED = PHEROMONE_DECAY ** 3  # 0.995^3 ~ 0.98507

    def decay(self):
        """Pheromone decay (compensated: called every 3rd tick)."""
        decay_factor = PheromoneMap._DECAY_COMPENSATED
        for row in self.grid:
            for cell in row:
                to_remove = []
                for ptype in cell:
                    cell[ptype] *= decay_factor
                    if cell[ptype] < 0.05:
                        to_remove.append(ptype)
                for k in to_remove:
                    del cell[k]


# --- Single-celled organism ---
class Cell:
    _id_counter = 0

    def __init__(self, x, y, genome=None, energy=60):
        Cell._id_counter += 1
        self.id = Cell._id_counter
        self.x = x
        self.y = y
        self.genome = genome or Genome()
        self.energy = energy
        self.age = 0
        self.alive = True
        self.angle = random.uniform(0, 2 * math.pi)  # Facing direction
        self.angular_vel = 0.0  # Angular velocity
        self.vx = 0.0
        self.vy = 0.0
        self.thrust = 0.0  # Current thrust (0-1)
        self.desired_angle = self.angle  # Where it wants to turn
        self.cilia_phase = random.uniform(0, 2 * math.pi)  # Cilia animation phase
        self.kills = 0
        self.children = 0
        self.generation = 0
        # Pack/herd
        self.target_id = None     # Current target (prey) id
        self.alert = False        # Received danger alert
        self.alert_x = 0.0       # Where the danger comes from
        self.alert_y = 0.0
        self.pack_mates = 0       # How many allies are nearby
        # Sprint system
        self.sprinting = False
        self.sprint_energy = 100.0  # Sprint reserve (0-100)
        self.sprint_cooldown = 0    # Cooldown countdown
        # --- Predator memory ---
        # Hunting grounds: locations of successful hunts (x, y, intensity)
        self.hunting_spots = []       # max 5 memories
        # Prey profile: what prey is easy? (defense, speed, size, pack_mates -> success?)
        self.prey_memory = []         # max 10 memories: (defense, speed, isolated, success)
        # Preference learned from memory (initially neutral)
        self.prefer_weak = 0.0        # Prefers weak prey (+) vs strong (-)
        self.prefer_isolated = 0.0    # Prefers isolated (+) vs herd (-)
        self.prefer_slow = 0.0        # Prefers slow (+) vs fast (-)
        # --- Herbivore memory ---
        # Pastures: where it previously found food (x, y, abundance)
        self.food_spots = []          # max 5 grazing spots
        self.last_food_x = 0.0       # Last eating location
        self.last_food_y = 0.0
        # Danger zones: where predators attacked (x, y, danger_level)
        self.danger_zones = []        # max 5 dangerous areas
        self.attacks_here = 0         # How many times attacked at this location
        self.migrating = False        # Currently migrating
        self.migrate_x = 0.0         # Migration target
        self.migrate_y = 0.0
        # --- Leadership system ---
        self.leader_id = None         # Who it follows (leader cell id)
        self.leadership = 0.0         # Leadership score (dynamically calculated)
        self.ticks_without_food = 0   # Ticks since last meal
        self.last_eat_tick = 0        # Last eating time
        # --- Camouflage / ambush ---
        self.hiding = False           # Hiding in shelter
        self.ambushing = False        # Ambushing (predator)
        self.ambush_ticks = 0         # How long in ambush
        self.in_shelter = False       # Whether inside a shelter
        self.eating_target = None     # Which food it's eating (idx)
        self.eating_progress = 0.0    # Eating progress
        # --- HP system ---
        self.max_hp = self.genome.size ** 2 * 0.5  # Bigger = more HP
        self.hp = self.max_hp
        self.hp_regen = 0.02 + self.genome.size * 0.003  # Slow regen
        self.damaged_flash = 0       # Damage visual effect
        # --- Juvenile period ---
        self.is_child = True          # Born as juvenile
        self.maturity = 0.4           # Starts at 40% size
        self.mature_age = 120 + int(self.genome.size * 8)  # Bigger ones grow longer
        # --- Combat ---
        self.attacking_target = None  # Who it's currently damaging
        self.being_attacked_by = []   # Who is attacking it
        # --- Reproduction ---
        self.seeking_mate = False     # Actively seeking a mate
        self.mate_target_id = None    # Chosen mate's id
        self.mate_cooldown = 0        # Post-reproduction cooldown
        # --- ROI hunting ---
        self.chase_energy_spent = 0.0  # Energy burned on pursuit
        self.chase_target_id = None    # Who it's currently chasing
        self.rest_after_hunt = 0       # Rest after unsuccessful hunt
        # --- Decision loyalty (anti-oscillation) ---
        self.decision_target_x = 0.0   # Current target X
        self.decision_target_y = 0.0   # Current target Y
        self.decision_cooldown = 0     # How many ticks to hold direction
        # --- Hibernation ---
        self.hibernating = False       # Spore/hibernation state
        self.has_hibernated = False    # Can hibernate once in lifetime
        self.hibernate_ticks = 0       # How long in hibernation
        # --- Bite shock ---
        self.stun_ticks = 0            # Stun (no movement, no AI)
        self.slow_ticks = 0            # Slow (reduced speed)
        self.slow_factor = 1.0         # Slow amount (0.3 = 30% speed)
        # --- Taunt system ---
        self.taunt_type = None         # Active taunt type: 'mate', 'attack', 'flee'
        self.taunt_timer = 0           # Taunt animation countdown
        self.taunt_target_id = None    # Attack taunt target id
        self.taunt_radius = 0.0        # Taunt wave current radius (visual)
        self.flee_zigzag_phase = 0.0   # Flee zigzag phase (unique frequency)

    @property
    def current_size(self):
        """Current size (smaller during juvenile period)."""
        if self.is_child:
            return self.genome.size * self.maturity
        return self.genome.size

    @property
    def radius(self):
        return self.current_size

    @property
    def hp_ratio(self):
        return self.hp / max(1, self.max_hp)

    @property
    def damage_per_tick(self):
        """Damage dealt: attack * size multiplier."""
        size_mult = self.current_size / 10.0  # size 10 = 1x, size 20 = 2x
        return self.genome.attack * size_mult * 0.3

    @property
    def alertness(self):
        """Well-fed prey becomes sluggish and less alert (1.0=hungry/alert, 0.65=full/lazy)."""
        if self.genome.is_predator():
            return 1.0  # Predators are always sharp
        fullness = min(self.energy / max(self.genome.repro_thresh, 1), 1.0)
        return 1.0 - fullness * 0.35  # 1.0 (starving) → 0.65 (stuffed)

    @property
    def effective_speed_mult(self):
        """Speed multiplier: damage + size slowdown + well-fed sluggishness."""
        hp_mult = max(0.25, self.hp_ratio)  # Half speed at half health
        child_bonus = 1.3 if self.is_child else 1.0  # Juveniles are faster
        return hp_mult * child_bonus * self.alertness

    def energy_cost_per_tick(self):
        # Hibernation: minimal metabolism (1%)
        if self.hibernating:
            return 0.002
        base = 0.018 if not self.genome.is_predator() else 0.02
        if self.genome.is_omnivore():
            base = 0.02  # Omnivore: medium
        size_cost = self.genome.size * 0.003
        # Cilia maintenance cost
        cilia_cost = self.genome.num_cilia * self.genome.cilia_power * 0.006
        # Active movement cost - almost zero when resting
        if self.thrust < 0.1:
            move_cost = 0.001  # Resting: almost free
        else:
            move_cost = self.thrust * self.genome.num_cilia * self.genome.cilia_power * 0.008
        # Sprint extra cost
        if self.sprinting:
            move_cost *= 2.5
        # Ambushing / hiding: minimal energy
        if self.ambushing or self.hiding:
            return base * AMBUSH_ENERGY_MULT + size_cost * 0.5
        # Predators: attacking is cheaper (specialized for it)
        if self.genome.is_predator():
            attack_cost = self.genome.attack * 0.001
        else:
            attack_cost = self.genome.attack * 0.004
        sense_cost = self.genome.sense_range * 0.00012
        # Stealth maintenance cost (predator specialization)
        stealth_cost = self.genome.stealth * 0.002 if self.genome.is_predator() else 0
        efficiency = self.genome.metabolism
        return (base + size_cost + cilia_cost + move_cost + attack_cost + sense_cost + stealth_cost) * efficiency

    def update(self, world_w, world_h, obstacles=None, energy_drain=1.0):
        if not self.alive:
            return

        self.age += 1
        self.energy -= self.energy_cost_per_tick() * energy_drain
        if self.mate_cooldown > 0:
            self.mate_cooldown -= 1
        if self.taunt_timer > 0:
            self.taunt_timer -= 1
            self.taunt_radius += 3.0 + self.genome.taunt_power * 3.0
            if self.taunt_timer <= 0:
                self.taunt_type = None
                self.taunt_target_id = None
        # Bite shock countdowns
        if self.stun_ticks > 0:
            self.stun_ticks -= 1
        if self.slow_ticks > 0:
            self.slow_ticks -= 1
            if self.slow_ticks <= 0:
                self.slow_factor = 1.0

        if self.energy <= MIN_CELL_ENERGY:
            # Predators and omnivores can hibernate instead of dying — BUT only once!
            if not self.hibernating and not self.has_hibernated and (self.genome.is_predator() or self.genome.is_omnivore()):
                self.hibernating = True
                self.has_hibernated = True  # Can hibernate once in lifetime
                self.hibernate_ticks = 0
                self.energy = 15.0  # Need reserve for waking up!
                self.thrust = 0.0
                self.sprinting = False
                self.vx = 0.0
                self.vy = 0.0
            else:
                self.alive = False
                return

        # Hibernation time limit: max ~500 ticks, then dies
        if self.hibernating:
            self.hibernate_ticks += 1
            if self.hibernate_ticks > 500:
                self.alive = False
                return

        # --- HP system ---
        if self.hp <= 0:
            self.alive = False
            return
        # HP regeneration (only when not in combat and has energy)
        if not self.being_attacked_by and self.hp < self.max_hp and self.energy > 20:
            self.hp = min(self.max_hp, self.hp + self.hp_regen)
        # being_attacked_by is reset at the start of World.update()
        if self.damaged_flash > 0:
            self.damaged_flash -= 1

        # --- Juvenile period: growth ---
        if self.is_child:
            growth_rate = 0.6 / self.mature_age  # Gradually grows
            self.maturity = min(1.0, self.maturity + growth_rate)
            if self.age >= self.mature_age:
                self.is_child = False
                self.maturity = 1.0
                # Adulthood: HP max update
                self.max_hp = self.genome.size ** 2 * 0.5
                self.hp = self.max_hp

        # Memory fading (every 200 ticks)
        if self.age % 200 == 0:
            self.danger_zones = [(x, y, d * 0.8) for x, y, d in self.danger_zones if d * 0.8 > 0.2]
            self.hunting_spots = [(x, y, i * 0.9) for x, y, i in self.hunting_spots if i * 0.9 > 0.1]
            self.food_spots = [(x, y, a * 0.95) for x, y, a in self.food_spots if a * 0.95 > 0.15]

        # Age asymmetry: predators live long (knowledge accumulation), herbivores short (r strategy)
        age_limit = 5000 if self.genome.is_predator() else 2500
        if self.age > age_limit:
            self.energy -= 0.04 * (self.age - age_limit) / 1000

        # --- Cilia-based movement ---
        # Rotation toward desired direction
        angle_diff = self.desired_angle - self.angle
        # Normalize to [-pi, pi]
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
        max_turn = self.genome.turn_rate
        # High spread = faster turning (jellyfish turns almost instantly)
        # spread=0: base turn_rate, spread=1: 4x faster
        effective_turn = max_turn * (1 + self.genome.maneuverability * 3)
        if abs(angle_diff) < effective_turn:
            self.angle = self.desired_angle
        else:
            self.angle += effective_turn if angle_diff > 0 else -effective_turn

        # --- Sprint handling ---
        # Larger cells drain sprint faster
        sprint_drain = 1.5 + self.genome.size * 0.15  # size 5=2.25, size 15=3.75, size 25=5.25
        sprint_recharge = max(0.1, 0.6 - self.genome.size * 0.02)  # Larger ones recharge slower
        if self.sprint_cooldown > 0:
            self.sprint_cooldown -= 1
        if self.sprinting:
            self.sprint_energy -= sprint_drain
            if self.sprint_energy <= 0:
                self.sprinting = False
                self.sprint_cooldown = int(40 + self.genome.size * 3)  # Larger ones rest longer
                self.sprint_energy = 0
        else:
            if self.sprint_cooldown <= 0:
                self.sprint_energy = min(100, self.sprint_energy + sprint_recharge)

        # Bite shock: stun -> no movement, slow -> reduced speed
        if self.stun_ticks > 0:
            self.thrust = 0.0
            self.vx *= 0.8  # Inertial deceleration
            self.vy *= 0.8
            self.sprinting = False
            # Position update (still slides a bit)
            self.x += self.vx
            self.y += self.vy
            self.cilia_phase += 0.02  # Minimal animation
            return  # Nothing else happens

        # Thrust calculation based on cilia configuration
        thrust = self.thrust
        # Apply slowdown
        if self.slow_ticks > 0:
            thrust *= self.slow_factor
        # Predators sprint harder (hunting rush)
        if self.sprinting:
            sprint_mult = 2.2 if self.genome.is_predator() else 1.6
        else:
            sprint_mult = 1.0
        cilia_angles = self.genome.cilia_positions()
        power = self.genome.cilia_power * sprint_mult * self.maturity  # Juvenile = weaker thrust
        spread = self.genome.maneuverability  # 0=torpedo, 1=jellyfish

        # Mix of two components:
        # 1) Torpedo force: cilia push in facing direction (traditional)
        # 2) Jellyfish force: cilia push in DESIRED direction (direct strafing)
        # The spread ratio determines which dominates

        # Torpedo component: each cilium pushes at its own angle (facing direction + offset)
        torpedo_fx, torpedo_fy = 0.0, 0.0
        for ca in cilia_angles:
            abs_angle = self.angle + ca
            torpedo_fx += math.cos(abs_angle) * power * thrust
            torpedo_fy += math.sin(abs_angle) * power * thrust

        # Jellyfish component: cilia coordinate in desired direction
        # More cilia = stronger strafing
        n_cilia = self.genome.num_cilia
        medusa_force = n_cilia * power * thrust
        medusa_fx = math.cos(self.desired_angle) * medusa_force
        medusa_fy = math.sin(self.desired_angle) * medusa_force

        # Mixing: spread=0 -> 100% torpedo, spread=1 -> 80% jellyfish + 20% torpedo
        fx = torpedo_fx * (1.0 - spread * 0.8) + medusa_fx * spread * 0.8
        fy = torpedo_fy * (1.0 - spread * 0.8) + medusa_fy * spread * 0.8

        # Apply force (bigger = heavier = slower acceleration)
        mass = self.current_size * 0.8  # Stronger size-slowdown
        self.vx += fx / mass
        self.vy += fy / mass

        # Friction (water resistance — bigger = more resistance)
        drag = 0.92 - self.current_size * 0.001  # size 20 -> 0.90
        drag = max(0.85, drag)
        self.vx *= drag
        self.vy *= drag

        # Speed limit: damage + juvenile + sprint
        speed_mult = self.effective_speed_mult
        max_spd = self.genome.max_speed * sprint_mult * speed_mult
        spd = math.sqrt(self.vx ** 2 + self.vy ** 2)
        if spd > max_spd:
            self.vx = self.vx / spd * max_spd
            self.vy = self.vy / spd * max_spd

        # Position update
        self.x += self.vx
        self.y += self.vy

        # Bounce off world edges
        if self.x < self.radius:
            self.x = self.radius
            self.vx *= -0.5
            self.angle = math.pi - self.angle
        elif self.x > world_w - self.radius:
            self.x = world_w - self.radius
            self.vx *= -0.5
            self.angle = math.pi - self.angle
        if self.y < self.radius:
            self.y = self.radius
            self.vy *= -0.5
            self.angle = -self.angle
        elif self.y > world_h - self.radius:
            self.y = world_h - self.radius
            self.vy *= -0.5
            self.angle = -self.angle

        # Obstacle collision — push out to nearest side
        if obstacles:
            r = self.radius
            for obs in obstacles:
                ox, oy, ow, oh = obs['x'], obs['y'], obs['w'], obs['h']
                # AABB overlap check
                if self.x + r > ox and self.x - r < ox + ow and \
                   self.y + r > oy and self.y - r < oy + oh:
                    # Which side is the nearest exit?
                    pen_left = (self.x + r) - ox
                    pen_right = (ox + ow) - (self.x - r)
                    pen_top = (self.y + r) - oy
                    pen_bottom = (oy + oh) - (self.y - r)
                    min_pen = min(pen_left, pen_right, pen_top, pen_bottom)
                    if min_pen == pen_left:
                        self.x = ox - r
                        self.vx *= -0.4
                    elif min_pen == pen_right:
                        self.x = ox + ow + r
                        self.vx *= -0.4
                    elif min_pen == pen_top:
                        self.y = oy - r
                        self.vy *= -0.4
                    else:
                        self.y = oy + oh + r
                        self.vy *= -0.4

        # Cilia animation
        self.cilia_phase += 0.15 * (0.5 + thrust)

    def steer_towards(self, tx, ty, thrust=0.8):
        """Set desired direction and thrust."""
        dx = tx - self.x
        dy = ty - self.y
        self.desired_angle = math.atan2(dy, dx)
        self.thrust = thrust

    def steer_away(self, tx, ty, thrust=0.9):
        dx = self.x - tx
        dy = self.y - ty
        self.desired_angle = math.atan2(dy, dx)
        self.thrust = thrust

    def wander(self):
        # If recently ate: slowly wander near the pasture
        if self.ticks_without_food < 60:
            self.desired_angle += random.gauss(0, 0.3)
            self.thrust = 0.1 + random.random() * 0.15
        else:
            # Hungry: search with larger angle changes, decisive direction
            # Occasionally takes a completely new direction (breaks out of circles)
            if random.random() < 0.08:
                self.desired_angle = random.uniform(0, 2 * math.pi)
            else:
                self.desired_angle += random.gauss(0, 0.5)
            self.thrust = 0.3 + random.random() * 0.3

    def remember_hunt(self, prey, success, x, y):
        """Remember the outcome of a hunt."""
        # Save prey profile
        prey_speed = prey.genome.max_speed
        prey_def = prey.genome.defense
        prey_isolated = 1.0 if prey.pack_mates < 2 else 0.0
        self.prey_memory.append((prey_def, prey_speed, prey_isolated, success))
        if len(self.prey_memory) > 10:
            self.prey_memory.pop(0)  # Remove oldest

        # Update preferences from experience
        if len(self.prey_memory) >= 3:
            successes = [m for m in self.prey_memory if m[3]]
            failures = [m for m in self.prey_memory if not m[3]]
            if successes:
                avg_s_def = sum(m[0] for m in successes) / len(successes)
                avg_s_spd = sum(m[1] for m in successes) / len(successes)
                avg_s_iso = sum(m[2] for m in successes) / len(successes)
                # Learning: prefers patterns from successful hunts
                self.prefer_weak += (3.0 - avg_s_def) * 0.1   # low def = easy
                self.prefer_slow += (2.0 - avg_s_spd) * 0.1   # slow = easy
                self.prefer_isolated += avg_s_iso * 0.15        # isolated = easy
            if failures:
                avg_f_def = sum(m[0] for m in failures) / len(failures)
                avg_f_iso = sum(m[2] for m in failures) / len(failures)
                # Avoids what didn't work
                self.prefer_weak += (avg_f_def - 3.0) * 0.05
                self.prefer_isolated -= (1.0 - avg_f_iso) * 0.05
            # Clipping
            self.prefer_weak = max(-1, min(2, self.prefer_weak))
            self.prefer_isolated = max(-1, min(2, self.prefer_isolated))
            self.prefer_slow = max(-1, min(2, self.prefer_slow))

        # Save hunting ground if successful
        if success:
            self.hunting_spots.append((x, y, 1.0))
            if len(self.hunting_spots) > 5:
                self.hunting_spots.pop(0)

    def best_hunting_spot(self):
        """Nearest known good hunting ground."""
        if not self.hunting_spots:
            return None
        # Best (most recent and strongest)
        best = max(self.hunting_spots, key=lambda s: s[2])
        return (best[0], best[1])

    def calc_fitness(self):
        """Fitness score: survival + reproduction + energy."""
        score = 0.0
        score += min(self.age / 300, 5.0)          # Age (max 5)
        score += self.children * 2.0                # Offspring (most important!)
        score += self.kills * 0.8                   # Kills (predator success)
        score += self.energy * 0.02                 # Current energy
        score += self.generation * 0.3              # Generation (evolutionary success)
        score += self.hp_ratio * 1.0                # Health
        return score

    def calc_leadership(self):
        """Leadership score: experience + survival + social + memory."""
        score = 0
        score += min(self.age / 500, 3.0)         # Age (max 3)
        score += self.kills * 0.5                   # Kills (predator experience)
        score += self.children * 0.3                # Reproduction (success)
        score += self.genome.social * 2.0           # Social tendency
        score += len(self.food_spots) * 0.3         # Pasture knowledge
        score += len(self.hunting_spots) * 0.4      # Hunting ground knowledge
        score += len(self.danger_zones) * 0.2       # Danger awareness
        score += self.energy * 0.01                 # Stronger = more reliable
        score += self.genome.taunt_power * 0.8       # Louder = better leader
        self.leadership = score
        return score

    def emit_taunt(self, taunt_type, target_id=None):
        """Emit taunt — sonar ping, range and speed depend on taunt_power gene."""
        tp = self.genome.taunt_power
        self.taunt_type = taunt_type
        base_timer = int(30 + tp * 15)  # Stronger gene = longer wave (30-67 ticks)
        # Predator mate calls travel 2x farther (sparse population, need wider reach)
        if taunt_type == 'mate' and self.genome.is_predator():
            base_timer *= 2
        self.taunt_timer = base_timer
        self.taunt_radius = 0.0
        self.taunt_target_id = target_id
        # Stronger taunt = more expensive (trade-off: energy vs range)
        self.energy -= 0.5 + tp * 1.5

    def remember_food(self, x, y):
        """Remember where food was found. Reinforces nearby spots."""
        self.last_food_x = x
        self.last_food_y = y
        # Near an existing spot?
        for i, (sx, sy, abundance) in enumerate(self.food_spots):
            if abs(sx - x) < 60 and abs(sy - y) < 60:
                # Reinforce the memory
                self.food_spots[i] = (sx, sy, min(abundance + 0.3, 3.0))
                return
        # New pasture discovered
        self.food_spots.append((x, y, 1.0))
        if len(self.food_spots) > 5:
            # Remove weakest
            self.food_spots.sort(key=lambda s: s[2], reverse=True)
            self.food_spots.pop()

    def remember_danger(self, pred_x, pred_y):
        """Remember where it was attacked — danger zone."""
        self.attacks_here += 1
        # Near an existing zone?
        for i, (dx, dy, level) in enumerate(self.danger_zones):
            if abs(dx - pred_x) < 80 and abs(dy - pred_y) < 80:
                self.danger_zones[i] = (dx, dy, min(level + 0.5, 5.0))
                return
        self.danger_zones.append((pred_x, pred_y, 1.0))
        if len(self.danger_zones) > 5:
            self.danger_zones.sort(key=lambda z: z[2], reverse=True)
            self.danger_zones.pop()

    def is_danger_zone(self, x, y):
        """How dangerous is this area (0-5)."""
        danger = 0
        for dx, dy, level in self.danger_zones:
            dist_sq = (dx - x) ** 2 + (dy - y) ** 2
            if dist_sq < 120 * 120:
                danger += level * (1.0 - math.sqrt(dist_sq) / 120)
        return danger

    def best_food_spot(self):
        """Best known pasture — also considers danger."""
        if not self.food_spots:
            return None
        # Scoring: richness - danger - proximity penalty (don't stay in place)
        best = None
        best_score = -999
        for sx, sy, abundance in self.food_spots:
            dist = math.sqrt((sx - self.x) ** 2 + (sy - self.y) ** 2)
            if dist < 20:
                continue  # Already here, don't send back
            danger = self.is_danger_zone(sx, sy)
            # Score: richer + safer + not too far
            score = abundance * 2.0 - danger * 1.5 - dist * 0.002
            if score > best_score:
                best_score = score
                best = (sx, sy)
        return best

    def should_migrate(self):
        """Decide whether to migrate: too many attacks or no food."""
        # 3+ attacks at same location -> migration
        if self.attacks_here >= 3:
            return True
        # Current location is dangerous
        if self.is_danger_zone(self.x, self.y) > 2.0:
            return True
        return False

    def start_migration(self):
        """Choose migration target — safe, rich pasture."""
        self.migrate_tries = getattr(self, 'migrate_tries', 0)
        target = self.best_food_spot()
        if target:
            self.migrate_x, self.migrate_y = target
            self.migrating = True
            self.attacks_here = 0
            self.migrate_tries = 0
            return True
        # No known pasture: random direction, farther each try
        angle = random.uniform(0, 2 * math.pi)
        dist = 300 + self.migrate_tries * 100  # Farther with each attempt
        self.migrate_x = self.x + math.cos(angle) * dist
        self.migrate_y = self.y + math.sin(angle) * dist
        self.migrating = True
        self.attacks_here = 0
        self.migrate_tries += 1
        return True

    def wants_to_mate(self):
        """Ready to seek a mate for reproduction."""
        if self.is_child or self.mate_cooldown > 0:
            return False
        # Herbivore/omnivore: DON'T reproduce while starving! (oasis depletion protection)
        if not self.genome.is_predator() and self.ticks_without_food > 30:
            return False
        thresh = self.genome.repro_thresh
        # Recently ate -> at pasture -> lower threshold
        if self.ticks_without_food < 30:
            thresh *= 0.7
        if self.age < 100 or self.energy < thresh:
            return False
        # Crowding-dependent
        if self.pack_mates > 6:
            extra = (self.pack_mates - 6) * 0.15
            return self.energy >= thresh * (1.0 + extra)
        return True

    def reproduce(self, partner=None, repro_cost=None):
        # K vs r strategy: predator = 1 strong offspring, herbivore = many fast offspring
        if self.genome.is_predator():
            litter = 1  # Predator always produces 1 strong offspring
            cost = self.energy * 0.65  # Gives more energy to offspring
        else:
            litter = max(2, int(self.genome.litter_size))
            cost = self.energy * (repro_cost if repro_cost else REPRODUCTION_COST_RATIO)
        self.energy -= cost
        # Partner energy contribution also goes to offspring (not destroyed!)
        partner_cost = 0.0
        if partner:
            partner_cost = partner.energy * (repro_cost if repro_cost else REPRODUCTION_COST_RATIO) * 0.3
            partner.energy -= partner_cost
            partner.children += litter
        total_energy = (cost + partner_cost) * 0.9  # 10% "birth loss"
        per_child_energy = total_energy / litter
        self.children += litter

        children = []
        for i in range(litter):
            if partner:
                # Fitness-weighted crossover: better parent's genes dominate
                self_fit = self.calc_fitness()
                partner_fit = partner.calc_fitness()
                child_genome = self.genome.crossover(
                    partner.genome, self_fit, partner_fit).mutate()
            else:
                # Parentless reproduction (asexual): mutation only
                child_genome = self.genome.mutate()

            # Offspring spread in a circle
            angle = (2 * math.pi * i / litter) + random.uniform(-0.3, 0.3)
            dist = self.radius * 3 + random.uniform(0, self.radius * 2)
            offset_x = math.cos(angle) * dist
            offset_y = math.sin(angle) * dist
            child = Cell(self.x + offset_x, self.y + offset_y, child_genome, per_child_energy)
            child.generation = self.generation + 1
            # Child HP = proportional to small size
            child.max_hp = (child.genome.size * child.maturity) ** 2 * 0.5
            child.hp = child.max_hp
            # Memory inheritance: better parent's memories are stronger
            # Which parent is dominant (based on fitness)?
            if partner:
                sf = self.calc_fitness()
                pf = partner.calc_fitness()
                dominant = self if sf >= pf else partner
                recessive = partner if sf >= pf else self
                dom_w, rec_w = 0.7, 0.3  # Dominant parent's memories are stronger
            else:
                dominant = self
                recessive = None
                dom_w, rec_w = 0.7, 0.0

            # Hunting grounds
            spots = [(x, y, iv * dom_w) for x, y, iv in dominant.hunting_spots]
            if recessive and recessive.hunting_spots:
                spots += [(x, y, iv * rec_w) for x, y, iv in recessive.hunting_spots]
            if spots:
                spots.sort(key=lambda s: s[2], reverse=True)
                child.hunting_spots = spots[:5]

            # Preferences: weighted average
            child.prefer_weak = dominant.prefer_weak * dom_w + (recessive.prefer_weak * rec_w if recessive else 0)
            child.prefer_isolated = dominant.prefer_isolated * dom_w + (recessive.prefer_isolated * rec_w if recessive else 0)
            child.prefer_slow = dominant.prefer_slow * dom_w + (recessive.prefer_slow * rec_w if recessive else 0)

            # Pastures
            food_sp = [(x, y, a * dom_w) for x, y, a in dominant.food_spots]
            if recessive and recessive.food_spots:
                food_sp += [(x, y, a * rec_w) for x, y, a in recessive.food_spots]
            if food_sp:
                food_sp.sort(key=lambda s: s[2], reverse=True)
                child.food_spots = food_sp[:5]

            # Danger zones: both parents' are important (survival!)
            danger = [(x, y, d * 0.6) for x, y, d in dominant.danger_zones]
            if recessive and recessive.danger_zones:
                danger += [(x, y, d * 0.5) for x, y, d in recessive.danger_zones]
            if danger:
                danger.sort(key=lambda s: s[2], reverse=True)
                child.danger_zones = danger[:5]
            children.append(child)
        return children


# --- Spatial grid for collision detection ---
class SpatialGrid:
    def __init__(self, cell_size=SPATIAL_GRID_SIZE):
        self.cell_size = cell_size
        self.grid = defaultdict(list)

    def clear(self):
        self.grid.clear()

    def insert(self, obj, x, y):
        gx = int(x // self.cell_size)
        gy = int(y // self.cell_size)
        self.grid[(gx, gy)].append(obj)

    def query(self, x, y, radius):
        """Returns all objects within the given radius."""
        results = []
        min_gx = int((x - radius) // self.cell_size)
        max_gx = int((x + radius) // self.cell_size)
        min_gy = int((y - radius) // self.cell_size)
        max_gy = int((y + radius) // self.cell_size)
        for gx in range(min_gx, max_gx + 1):
            for gy in range(min_gy, max_gy + 1):
                results.extend(self.grid.get((gx, gy), []))
        return results


# --- World ---
class World:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.cells = []
        self.food = []  # [(x, y, energy), ...]
        self.tick = 0
        self.cell_grid = SpatialGrid()
        self.food_grid = SpatialGrid()
        self.stats = {
            "total_born": 0,
            "total_died": 0,
            "total_eaten": 0,
            "max_generation": 0,
            "predators": 0,
            "herbivores": 0,
            "omnivores": 0,
        }
        # Live settings (modifiable from menu)
        self.settings = {
            'food_energy': FOOD_ENERGY,
            'eat_speed': EAT_SPEED,
            'food_regrow': 0.06,            # Food regrowth rate
            'repro_cost': REPRODUCTION_COST_RATIO,
            'mutation_rate': MUTATION_RATE,
            'mutation_strength': MUTATION_STRENGTH,
            'max_pop': 400,
            'energy_drain': 1.0,
            'pred_sense': 1.0,
        }
        self._init()

    def _init(self):
        self.cells = []
        self.food = []
        self.tick = 0
        Cell._id_counter = 0

        # Pheromone map
        self.pheromones = PheromoneMap(self.width, self.height)

        # --- Fixed food sources (oases) ---
        # These are permanent locations where plants grow, like pastures
        self.food_sources = []
        for _ in range(INITIAL_FOOD):
            cx = random.uniform(150, self.width - 150)
            cy = random.uniform(150, self.height - 150)
            richness = random.uniform(0.6, 1.0)  # How fertile
            spread = random.uniform(60, 120)  # Larger spread, easier to find
            self.food_sources.append({
                'x': cx, 'y': cy,
                'richness': richness,
                'spread': spread,
                'capacity': int(25 + richness * 30),  # Max food that can grow here
            })
        # Shelters: half near oases (strategic), half random
        self.shelters = []
        for i in range(NUM_SHELTERS):
            if i < len(self.food_sources) and i < NUM_SHELTERS // 2:
                # Near oasis (ambush position!)
                src = self.food_sources[i]
                angle = random.uniform(0, 2 * math.pi)
                dist = src['spread'] * 1.5 + random.uniform(20, 60)
                sx = src['x'] + math.cos(angle) * dist
                sy = src['y'] + math.sin(angle) * dist
            else:
                sx = random.uniform(100, self.width - 100)
                sy = random.uniform(100, self.height - 100)
            sx = max(50, min(self.width - 50, sx))
            sy = max(50, min(self.height - 50, sy))
            r = random.uniform(SHELTER_RADIUS * 0.7, SHELTER_RADIUS * 1.3)
            self.shelters.append({'x': sx, 'y': sy, 'radius': r})

        # Obstacles (rocks) — BETWEEN oases, not on them
        self.obstacles = []
        for _ in range(NUM_OBSTACLES):
            for _try in range(20):  # Max 20 attempts to find a good spot
                ox = random.uniform(120, self.width - 120)
                oy = random.uniform(120, self.height - 120)
                ow = random.uniform(OBSTACLE_MIN_SIZE, OBSTACLE_MAX_SIZE)
                oh = random.uniform(OBSTACLE_MIN_SIZE, OBSTACLE_MAX_SIZE)
                # Don't place on oasis center
                too_close = False
                for src in self.food_sources:
                    dx = src['x'] - (ox + ow / 2)
                    dy = src['y'] - (oy + oh / 2)
                    if dx * dx + dy * dy < (src['spread'] + 60) ** 2:
                        too_close = True
                        break
                if not too_close:
                    self.obstacles.append({
                        'x': ox, 'y': oy, 'w': ow, 'h': oh,
                        # Visual variation
                        'shade': random.uniform(0.7, 1.0),
                        'jagged': [random.uniform(0.85, 1.15) for _ in range(8)],
                    })
                    break

        # Starting food: from oases
        for src in self.food_sources:
            for _ in range(src['capacity'] // 2):
                x = src['x'] + random.gauss(0, src['spread'])
                y = src['y'] + random.gauss(0, src['spread'])
                if 10 < x < self.width - 10 and 10 < y < self.height - 10:
                    food_e = self.settings['food_energy'] * random.uniform(0.4, 1.8)
                    self.food.append((x, y, food_e, False))

        # Population history (for graph)
        self.pop_history = []  # [(tick, herb, omni, pred), ...]
        self.pop_history_interval = 30  # Sample every 30 ticks

        # Starting cells
        for _ in range(INITIAL_CELLS):
            x = random.uniform(50, self.width - 50)
            y = random.uniform(50, self.height - 50)
            cell = Cell(x, y, energy=80)
            self.cells.append(cell)
            self.stats["total_born"] += 1

    def calc_detection(self, prey, predator, dist):
        """Prey's detection chance of predator: 0.0 = invisible, 1.0 = fully visible.
        Depends on predator's speed, distance, stealth gene, hiding."""
        sense = prey.genome.sense_range

        # 1) Predator's current speed vs max speed -> stealth level
        spd = math.sqrt(predator.vx * predator.vx + predator.vy * predator.vy)
        max_spd = max(predator.genome.max_speed * predator.effective_speed_mult, 0.1)
        speed_ratio = min(spd / max_spd, 1.0)  # 0=stationary, 1=max speed

        # Fast movement = easily noticed (base_visibility high)
        # Slow = hard (base_visibility low)
        base_visibility = speed_ratio  # 0->1

        # 2) Stealth gene reduces base detectability
        gene_stealth = predator.genome.stealth * 0.35  # Max 0.35 bonus

        # 3) Hiding/ambushing further reduction
        hide_bonus = 0.25 if (predator.ambushing or predator.hiding) else 0.0

        # Total stealth (cap 0.95 — always minimal chance)
        stealth = min(0.95, (1.0 - base_visibility) * 0.6 + gene_stealth + hide_bonus)

        # 4) Distance factor: closer = easier to detect
        #    dist/sense = 0 -> close_factor = 1.0 (stealth doesn't help)
        #    dist/sense = 1 -> close_factor = 0.0 (stealth maximally effective)
        normalized = min(dist / max(sense, 1), 1.0)
        close_factor = 1.0 - normalized  # 1.0 up close, 0.0 from far

        # Final visibility: stealth is full from afar, doesn't help up close
        visibility = close_factor + (1.0 - close_factor) * (1.0 - stealth)

        return visibility

    def has_line_of_sight(self, x1, y1, x2, y2):
        """Is there clear line of sight between two points? (Obstacles block it.)
        Liang-Barsky line-rectangle intersection."""
        # Quick bounding box: the line's bounding box
        min_x = min(x1, x2)
        max_x = max(x1, x2)
        min_y = min(y1, y2)
        max_y = max(y1, y2)
        for obs in self.obstacles:
            # Obstacle rectangle corners
            ox, oy = obs['x'], obs['y']
            ox2, oy2 = ox + obs['w'], oy + obs['h']
            # Quick exclusion: if line BB doesn't overlap obstacle
            if max_x < ox or min_x > ox2 or max_y < oy or min_y > oy2:
                continue
            # Liang-Barsky line-rectangle intersection
            dx = x2 - x1
            dy = y2 - y1
            p = [-dx, dx, -dy, dy]
            q = [x1 - ox, ox2 - x1, y1 - oy, oy2 - y1]
            t0, t1 = 0.0, 1.0
            valid = True
            for i in range(4):
                if abs(p[i]) < 1e-10:
                    if q[i] < 0:
                        valid = False
                        break
                else:
                    t = q[i] / p[i]
                    if p[i] < 0:
                        t0 = max(t0, t)
                    else:
                        t1 = min(t1, t)
                    if t0 > t1:
                        valid = False
                        break
            if valid and t0 <= t1:
                return False  # Obstacle blocks it
        return True  # Clear line of sight

    def _spawn_food_cluster(self):
        """Regrow food in a random oasis."""
        if not self.food_sources:
            return
        src = random.choice(self.food_sources)
        # How much food is currently in this oasis
        nearby_food = sum(1 for f in self.food
                         if abs(f[0] - src['x']) < src['spread'] * 2
                         and abs(f[1] - src['y']) < src['spread'] * 2)
        to_spawn = min(random.randint(3, 8), src['capacity'] - nearby_food)
        for _ in range(max(0, to_spawn)):
            x = src['x'] + random.gauss(0, src['spread'])
            y = src['y'] + random.gauss(0, src['spread'])
            if 10 < x < self.width - 10 and 10 < y < self.height - 10 and len(self.food) < MAX_FOOD:
                food_e = self.settings['food_energy'] * random.uniform(0.4, 1.8)
                self.food.append((x, y, food_e, False))

    def save_to_json(self, filepath="cell_save.json"):
        """Save world state to JSON file."""
        data = {
            'version': 1,
            'tick': self.tick,
            'width': self.width,
            'height': self.height,
            'settings': self.settings,
            'stats': self.stats,
            'food_sources': self.food_sources,
            'shelters': self.shelters,
            'obstacles': self.obstacles,
            'food': [(f[0], f[1], f[2], f[3] if len(f) > 3 else False,
                      f[4] if len(f) > 4 else 0) for f in self.food],
            'pop_history': self.pop_history,
            'cells': [],
        }
        for c in self.cells:
            if not c.alive:
                continue
            cell_data = {
                'genes': c.genome.genes.tolist(),
                'x': c.x, 'y': c.y,
                'energy': c.energy,
                'age': c.age,
                'generation': c.generation,
                'children': c.children,
                'kills': c.kills,
                'hp': c.hp,
                'max_hp': c.max_hp,
                'angle': c.angle,
                'is_child': c.is_child,
                'maturity': c.maturity,
                'has_hibernated': c.has_hibernated,
                'ticks_without_food': c.ticks_without_food,
                'last_eat_tick': c.last_eat_tick,
                'hunting_spots': c.hunting_spots,
                'food_spots': c.food_spots,
                'danger_zones': c.danger_zones,
                'prefer_weak': c.prefer_weak,
                'prefer_isolated': c.prefer_isolated,
                'prefer_slow': c.prefer_slow,
            }
            data['cells'].append(cell_data)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        return len(data['cells'])

    def load_from_json(self, filepath="cell_save.json"):
        """Load world state from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # World size + tick
        self.tick = data.get('tick', 0)
        self.settings = data.get('settings', self.settings)
        self.stats = data.get('stats', self.stats)
        self.pop_history = data.get('pop_history', [])

        # Oases and shelters
        self.food_sources = data.get('food_sources', self.food_sources)
        self.shelters = data.get('shelters', self.shelters)
        self.obstacles = data.get('obstacles', self.obstacles)

        # Food reload
        self.food = []
        for fd in data.get('food', []):
            if len(fd) >= 5:
                self.food.append((fd[0], fd[1], fd[2], fd[3], fd[4]))
            elif len(fd) >= 4:
                self.food.append((fd[0], fd[1], fd[2], fd[3]))
            else:
                self.food.append((fd[0], fd[1], fd[2], False))

        # Pheromone map reset
        self.pheromones = PheromoneMap(self.width, self.height)

        # Cells reload
        self.cells = []
        Cell._id_counter = 0
        for cd in data.get('cells', []):
            genome = Genome(genes=cd['genes'])
            cell = Cell(cd['x'], cd['y'], genome=genome, energy=cd['energy'])
            cell.age = cd.get('age', 0)
            cell.generation = cd.get('generation', 0)
            cell.children = cd.get('children', 0)
            cell.kills = cd.get('kills', 0)
            cell.hp = cd.get('hp', cell.max_hp)
            cell.max_hp = cd.get('max_hp', cell.max_hp)
            cell.angle = cd.get('angle', cell.angle)
            cell.is_child = cd.get('is_child', False)
            cell.maturity = cd.get('maturity', 1.0)
            cell.has_hibernated = cd.get('has_hibernated', False)
            cell.ticks_without_food = cd.get('ticks_without_food', 0)
            cell.last_eat_tick = cd.get('last_eat_tick', 0)
            cell.hunting_spots = cd.get('hunting_spots', [])
            cell.food_spots = cd.get('food_spots', [])
            cell.danger_zones = cd.get('danger_zones', [])
            cell.prefer_weak = cd.get('prefer_weak', 0.0)
            cell.prefer_isolated = cd.get('prefer_isolated', 0.0)
            cell.prefer_slow = cd.get('prefer_slow', 0.0)
            self.cells.append(cell)

        # Spatial grid rebuild
        self.cell_grid = SpatialGrid()
        self.food_grid = SpatialGrid()
        for c in self.cells:
            self.cell_grid.insert(c, c.x, c.y)
        for i, f in enumerate(self.food):
            self.food_grid.insert(i, f[0], f[1])

        return len(self.cells)

    def update(self):
        self.tick += 1

        # Reset attack memory before any AI runs (prevents execution order desync)
        for cell in self.cells:
            cell.being_attacked_by = []

        # Pheromone decay
        if self.tick % 3 == 0:  # Not every tick, CPU saving
            self.pheromones.decay()

        # Corpse scent: carcasses emit growing stench (CPU-friendly)
        if self.tick % 10 == 0:
            corpse_count = 0
            for f in self.food:
                if len(f) > 4 and f[3]:  # is_meat + has birth_tick
                    corpse_age = self.tick - f[4]
                    if corpse_age < 0:
                        continue
                    cloud_radius = min(5, 1 + corpse_age // 500)
                    intensity = min(2.5, 0.3 + f[2] * 0.015 + corpse_age * 0.0004)
                    self.pheromones.deposit_cloud(
                        f[0], f[1], PheromoneMap.CORPSE_SCENT,
                        intensity, cloud_radius)
                    corpse_count += 1
                    if corpse_count >= 20:  # Max 20 corpses/tick (CPU limit)
                        break

        # Food: regrows in oases
        if len(self.food) < MAX_FOOD:
            for src in self.food_sources:
                # Base regrowth: continuous
                if random.random() < self.settings['food_regrow'] * src['richness']:
                    # Spatial grid query instead of full list scan
                    nearby_food = len(self.food_grid.query(src['x'], src['y'], src['spread'] * 2))
                    if nearby_food < src['capacity']:
                        # Burst spawn: fewer food = faster regrowth
                        if nearby_food < src['capacity'] * 0.3:
                            burst = random.randint(2, 4)  # Very empty: fast regeneration
                        elif nearby_food < src['capacity'] * 0.6:
                            burst = random.randint(1, 3)
                        else:
                            burst = 1
                        for _ in range(burst):
                            x = src['x'] + random.gauss(0, src['spread'])
                            y = src['y'] + random.gauss(0, src['spread'])
                            if 10 < x < self.width - 10 and 10 < y < self.height - 10 and len(self.food) < MAX_FOOD:
                                # Variable sized food: small grass -> large bush
                                base_e = self.settings['food_energy']
                                food_e = base_e * random.uniform(0.4, 1.8)
                                self.food.append((x, y, food_e, False))
                                self.pheromones.deposit(x, y, PheromoneMap.FOOD_HERE, 0.3)

        # Spatial grid refresh
        self.cell_grid.clear()
        self.food_grid.clear()
        for cell in self.cells:
            if cell.alive:
                self.cell_grid.insert(cell, cell.x, cell.y)
        for i, food_item in enumerate(self.food):
            self.food_grid.insert(i, food_item[0], food_item[1])

        # AI and movement
        new_cells = []
        for cell in self.cells:
            if not cell.alive:
                continue
            # Shelter check
            cell.in_shelter = self.is_in_shelter(cell.x, cell.y)
            # Pheromone trails
            if self.tick % 5 == 0:
                if cell.genome.is_predator():
                    self.pheromones.deposit(cell.x, cell.y, PheromoneMap.DANGER, 0.2)
                elif cell.genome.is_omnivore():
                    self.pheromones.deposit(cell.x, cell.y, PheromoneMap.TRAIL, 0.12)
                    self.pheromones.deposit(cell.x, cell.y, PheromoneMap.PREY_SCENT, 0.15)  # Weaker prey scent
                else:
                    self.pheromones.deposit(cell.x, cell.y, PheromoneMap.TRAIL, 0.15)
                    # Herbivores leave prey scent (detected by predators)
                    self.pheromones.deposit(cell.x, cell.y, PheromoneMap.PREY_SCENT, 0.25)
                # Mating pheromone (weak base scent)
                if cell.seeking_mate:
                    self.pheromones.deposit(cell.x, cell.y, PheromoneMap.MATE, 0.3)
            # Mating pheromone CLOUD: every tick, drastically strong!
            # This is the main mechanism for allies to find each other
            if cell.seeking_mate:
                # The longer it searches, the stronger and larger the cloud
                mate_ticks = getattr(cell, 'mate_search_ticks', 0) + 1
                cell.mate_search_ticks = mate_ticks
                cloud_radius = min(6, 2 + mate_ticks // 60)  # 2->6 grid radius (80->240px)
                cloud_intensity = min(2.5, 0.5 + mate_ticks * 0.008)
                if self.tick % 3 == 0:  # Deposit cloud every 3 ticks (CPU saving)
                    self.pheromones.deposit_cloud(
                        cell.x, cell.y, PheromoneMap.MATE,
                        cloud_intensity, cloud_radius)
            else:
                cell.mate_search_ticks = 0
            self._cell_ai(cell)

            # --- Overcrowding penalty ---
            crowding = self.cell_grid.query(cell.x, cell.y, OVERCROWDING_RADIUS)
            # Exact circle filter (SpatialGrid returns a square)
            r_sq = OVERCROWDING_RADIUS * OVERCROWDING_RADIUS
            crowd_count = sum(1 for o in crowding
                              if o.id != cell.id and
                              (o.x - cell.x) ** 2 + (o.y - cell.y) ** 2 < r_sq)
            if crowd_count > OVERCROWDING_THRESHOLD:
                excess = crowd_count - OVERCROWDING_THRESHOLD
                cell.energy -= excess * OVERCROWDING_PENALTY

            cell.update(self.width, self.height, self.obstacles, self.settings.get('energy_drain', 1.0))

            if not cell.alive:
                # Corpse -> carcass (size-proportional energy) — 5th element: birth tick (freshness)
                corpse_energy = cell.energy * 0.5 + cell.genome.size ** 2 * 0.8
                corpse_energy = max(5, corpse_energy)
                self.food.append((cell.x, cell.y, corpse_energy, True, self.tick))
                self.stats["total_died"] += 1
                continue

            # Everyone eats - but based on type: herbivore=plant, predator=meat
            self._eat_food(cell)

            # Reproduction: mate-seeking required!
            if cell.wants_to_mate():
                cell.seeking_mate = True
                # Look for a mate nearby — both must be ready
                mate_range = cell.radius * 4 + 12  # Mating distance
                partner = None
                nearby = self.cell_grid.query(cell.x, cell.y, mate_range)
                for other in nearby:
                    if other.id == cell.id or not other.alive:
                        continue
                    # Reproductive isolation: only same species can mate
                    same_type = abs(cell.genome.diet - other.genome.diet) < 0.15
                    if same_type and other.wants_to_mate():
                        dx = other.x - cell.x
                        dy = other.y - cell.y
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < mate_range:
                            partner = other
                            break
                if partner:
                    # Both reproduce, partner also gets cooldown
                    children = cell.reproduce(partner, repro_cost=self.settings['repro_cost'])
                    cell.mate_cooldown = 200  # ~3 seconds cooldown
                    partner.mate_cooldown = 200
                    cell.seeking_mate = False
                    partner.seeking_mate = False
                    # Partner energy deduction happens in reproduce(), not here!
                    for child in children:
                        child.x = np.clip(child.x, 5, self.width - 5)
                        child.y = np.clip(child.y, 5, self.height - 5)
                        new_cells.append(child)
                        self.stats["total_born"] += 1
                        self.stats["max_generation"] = max(self.stats["max_generation"], child.generation)
            else:
                cell.seeking_mate = False

        # Predator attacks
        for cell in self.cells:
            if cell.alive and cell.genome.is_predator() and not cell.hibernating:
                self._predator_attack(cell)

        # Remove dead, add new
        self.cells = [c for c in self.cells if c.alive]
        # Population limit - most populous species can't reproduce when cap reached
        max_pop = self.settings.get('max_pop', 400)
        if max_pop > 0 and len(self.cells) + len(new_cells) > max_pop:
            # Count species
            herb_count = sum(1 for c in self.cells if c.genome.diet < 0.3)
            omni_count = sum(1 for c in self.cells if 0.3 <= c.genome.diet < 0.7)
            pred_count = sum(1 for c in self.cells if c.genome.diet >= 0.7)
            biggest = max(herb_count, omni_count, pred_count)
            # Block offspring from the most populous species
            if biggest == herb_count:
                blocked_diet = lambda d: d < 0.3
            elif biggest == pred_count:
                blocked_diet = lambda d: d >= 0.7
            else:
                blocked_diet = lambda d: 0.3 <= d < 0.7
            new_cells = [c for c in new_cells if not blocked_diet(c.genome.diet)]
            # Hard cap: trim remaining if still over
            if len(self.cells) + len(new_cells) > max_pop:
                new_cells = new_cells[:max(0, max_pop - len(self.cells))]
        self.cells.extend(new_cells)

        # --- Cell-cell collision (bounding circle) ---
        # Rebuild spatial grid for collision queries
        self.cell_grid.clear()
        for cell in self.cells:
            if cell.alive:
                self.cell_grid.insert(cell, cell.x, cell.y)
        # Push apart overlapping cells
        for cell in self.cells:
            if not cell.alive:
                continue
            nearby = self.cell_grid.query(cell.x, cell.y, cell.radius * 4)
            for other in nearby:
                if other.id <= cell.id or not other.alive:
                    continue  # Each pair only once (id ordering)
                # No collision between predator and its prey (would prevent biting)
                cell_pred = cell.genome.is_predator()
                other_pred = other.genome.is_predator()
                if cell_pred != other_pred:
                    continue  # Predator vs non-predator: no push
                dx = other.x - cell.x
                dy = other.y - cell.y
                dist_sq = dx * dx + dy * dy
                min_dist = cell.radius + other.radius
                if dist_sq < min_dist * min_dist and dist_sq > 0.01:
                    dist = math.sqrt(dist_sq)
                    overlap = min_dist - dist
                    # Push force: heavier cells move less
                    total_mass = cell.current_size + other.current_size
                    cell_ratio = other.current_size / total_mass  # Lighter cell moves more
                    other_ratio = cell.current_size / total_mass
                    nx = dx / dist  # Normal vector
                    ny = dy / dist
                    push = overlap * 0.5  # Resolve half per tick (soft collision)
                    cell.x -= nx * push * cell_ratio
                    cell.y -= ny * push * cell_ratio
                    other.x += nx * push * other_ratio
                    other.y += ny * push * other_ratio
                    # Dampen velocity along collision normal
                    cell.vx -= nx * push * cell_ratio * 0.3
                    cell.vy -= ny * push * cell_ratio * 0.3
                    other.vx += nx * push * other_ratio * 0.3
                    other.vy += ny * push * other_ratio * 0.3

        # Food cleanup: remove eaten food (energy <= 0) all at once at end of tick
        # This prevents index-shift bugs (phantom food)
        self.food = [f for f in self.food if f[2] > FOOD_MIN_ENERGY]

        # Statistics update
        self.stats["predators"] = sum(1 for c in self.cells if c.genome.is_predator())
        self.stats["omnivores"] = sum(1 for c in self.cells if c.genome.is_omnivore())
        self.stats["herbivores"] = len(self.cells) - self.stats["predators"] - self.stats["omnivores"]

        # Population history recording (for graph)
        if self.tick % self.pop_history_interval == 0:
            self.pop_history.append((
                self.tick,
                self.stats["herbivores"],
                self.stats["omnivores"],
                self.stats["predators"],
            ))
            # Max 600 sample points (last ~18000 ticks)
            if len(self.pop_history) > 600:
                self.pop_history = self.pop_history[-600:]

    def spawn_herbivore(self, x, y):
        """Manual herbivore spawn."""
        g = Genome()
        g.genes[5] = random.uniform(0.0, 0.3)   # diet: herbivore
        g.genes[1] = random.uniform(150, 280)    # sense_range: good perception
        g.genes[3] = random.uniform(2, 5)        # defense
        g.genes[6] = random.uniform(35, 70)      # repro_thresh: low threshold
        g.genes[9] = random.uniform(0.3, 0.7)    # social
        g.genes[14] = random.uniform(1.5, 3)     # litter_size: more offspring
        c = Cell(x, y, g, energy=100)
        self.cells.append(c)
        self.stats["total_born"] += 1

    def spawn_predator(self, x, y):
        """Manual predator spawn."""
        g = Genome()
        g.genes[5] = random.uniform(0.75, 1.0)
        g.genes[2] = random.uniform(3, 6)
        g.genes[1] = random.uniform(100, 200)
        g.genes[9] = random.uniform(0.3, 0.7)
        g.genes[6] = random.uniform(30, 55)     # repro_thresh: lower
        c = Cell(x, y, g, energy=160)
        self.cells.append(c)
        self.stats["total_born"] += 1

    def spawn_omnivore(self, x, y):
        """Manual omnivore spawn."""
        g = Genome()
        g.genes[5] = random.uniform(0.35, 0.65)   # diet: omnivore range
        g.genes[1] = random.uniform(120, 230)      # sense_range
        g.genes[2] = random.uniform(1, 3)          # attack: low
        g.genes[3] = random.uniform(2, 5)          # defense: medium
        g.genes[6] = random.uniform(40, 80)        # repro_thresh
        g.genes[9] = random.uniform(0.2, 0.6)      # social
        g.genes[14] = random.uniform(1, 2.5)       # litter_size
        c = Cell(x, y, g, energy=100)
        self.cells.append(c)
        self.stats["total_born"] += 1

    def is_in_shelter(self, x, y):
        """Whether inside a shelter."""
        for s in self.shelters:
            dx = x - s['x']
            dy = y - s['y']
            if dx * dx + dy * dy < s['radius'] * s['radius']:
                return True
        return False

    def _cell_ai(self, cell):
        """Cell behavior with pack/herd mechanics + leadership system."""
        # Stunned cell: no decisions, no movement
        if cell.stun_ticks > 0:
            return
        # Hibernating cell: does nothing, just waits for prey
        if cell.hibernating:
            cell.thrust = 0.0
            cell.vx = 0.0
            cell.vy = 0.0
            # Wake up: prey within sense range!
            sense = cell.genome.sense_range
            nearby_cells = self.cell_grid.query(cell.x, cell.y, sense)
            for other in nearby_cells:
                if other.id == cell.id or not other.alive:
                    continue
                if other.genome.is_predator():
                    continue  # Predator doesn't wake it up
                dx = other.x - cell.x
                dy = other.y - cell.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < sense:
                    # WAKE UP! Adrenaline rush at the cost of HP!
                    cell.hibernating = False
                    cell.sprint_energy = 60
                    cell.sprinting = True
                    cell.hp -= cell.max_hp * 0.15     # Muscle atrophy cost for the final rush
                    cell.steer_towards(other.x, other.y, 1.0)
                    break
            # Omnivore hibernating: food also wakes it up
            if cell.hibernating and cell.genome.is_omnivore():
                nearby_food = self.food_grid.query(cell.x, cell.y, sense * 0.5)
                if nearby_food:
                    cell.hibernating = False
            return  # No other AI during hibernation

        sense = cell.genome.sense_range
        # Well-fed prey: reduced awareness (alertness 0.65-1.0)
        if not cell.genome.is_predator():
            sense *= cell.alertness
        else:
            sense *= self.settings.get('pred_sense', 1.0)
        nearby_cells = self.cell_grid.query(cell.x, cell.y, sense)

        # Starvation counter (no starvation in first 300 ticks, has starting energy)
        if cell.last_eat_tick > 0:
            cell.ticks_without_food = self.tick - cell.last_eat_tick
        elif cell.age > 300:
            cell.ticks_without_food = cell.age - 300
        else:
            cell.ticks_without_food = 0

        # Separate allies and enemies + leader search
        allies = []
        enemies = []
        cell.pack_mates = 0
        best_leader = None
        best_leader_score = cell.calc_leadership()  # Own leadership

        for other in nearby_cells:
            if other.id == cell.id or not other.alive:
                continue
            dx = other.x - cell.x
            dy = other.y - cell.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > sense:
                continue
            # Line of sight check — can't see behind obstacles
            if not self.has_line_of_sight(cell.x, cell.y, other.x, other.y):
                continue
            # Alliance determination with species-level logic:
            # Predator + predator = allies
            # Herbivore + herbivore = allies
            # Omnivore + herbivore = allies (mixed herd)
            # Omnivore + omnivore = allies
            # Predator + anyone else = enemies
            cell_pred = cell.genome.is_predator()
            other_pred = other.genome.is_predator()
            if cell_pred and other_pred:
                is_same_type = True
            elif cell_pred or other_pred:
                is_same_type = False  # Predator vs non-predator = enemy
            else:
                is_same_type = True   # Herbivore/omnivore = allies
            if is_same_type:
                allies.append((other, dist))
                cell.pack_mates += 1
                # Leader search: more experienced + more social = better leader
                if cell.genome.social > 0.3:
                    other_score = other.calc_leadership()
                    if other_score > best_leader_score:
                        best_leader = other
                        best_leader_score = other_score
            else:
                enemies.append((other, dist))

        # Leader update
        cell.leader_id = best_leader.id if best_leader else None

        # === TAUNT SYSTEM ===
        is_alpha = best_leader is None and cell.pack_mates > 0

        # --- Taunt EMISSION (taunt_power gene determines activity) ---
        tp = cell.genome.taunt_power
        if cell.taunt_timer <= 0 and tp > 0.1 and not cell.is_child:
            # Frequency based on gene strength: stronger gene = more frequent taunting
            taunt_interval = max(20, int(80 - tp * 30))  # Between 20-77 ticks

            # 1) MATE TAUNT: seeking mate and no one nearby
            if cell.seeking_mate and not any(a.wants_to_mate() for a, _ in allies):
                if cell.age % taunt_interval == 0:
                    cell.emit_taunt('mate')

            # 2) ATTACK TAUNT: alpha predator sends pack to target
            elif is_alpha and cell.genome.is_predator() and cell.target_id is not None:
                if cell.genome.aggression > 0.3 and cell.pack_mates >= 2:
                    if cell.age % taunt_interval == 0:
                        cell.emit_taunt('attack', cell.target_id)

            # 3) FLEE TAUNT: being attacked -> panic signal (immediate, doesn't wait for interval)
            elif cell.being_attacked_by and cell.genome.social > 0.15:
                cell.emit_taunt('flee')

        # --- Taunt RECEPTION: reacting to nearby taunts ---
        if allies and cell.genome.social > 0.15:
            for ally, dist in allies:
                if ally.taunt_timer <= 0:
                    continue
                # Wavefront: taunt only affects ONCE, when the edge passes over the cell
                wave_speed = 3.0 + ally.genome.taunt_power * 3.0
                if not (ally.taunt_radius - wave_speed <= dist <= ally.taunt_radius):
                    continue

                if ally.taunt_type == 'mate':
                    # Mating call -> if ready to mate, sprint toward caller
                    if cell.wants_to_mate() and not cell.seeking_mate:
                        cell.seeking_mate = True
                        cell.mate_target_id = ally.id
                    if cell.seeking_mate:
                        cell.steer_towards(ally.x, ally.y, 0.8)
                        if dist > 50 and cell.sprint_energy > 20 and cell.sprint_cooldown <= 0:
                            cell.sprinting = True

                elif ally.taunt_type == 'attack' and ally.taunt_target_id is not None:
                    # Alpha attack command -> take over target
                    if cell.genome.is_predator():
                        cell.target_id = ally.taunt_target_id
                        # Search: is the target nearby?
                        for other, odist in enemies:
                            if other.id == ally.taunt_target_id:
                                cell.steer_towards(other.x, other.y, 0.9)
                                if odist < sense * 0.4 and cell.sprint_energy > 15:
                                    cell.sprinting = True
                                break

                elif ally.taunt_type == 'flee':
                    # Panic signal -> flee from danger direction, with zigzag
                    if not cell.genome.is_predator():
                        # Danger source is the position of attackers on the taunt emitter
                        flee_from_x = ally.x
                        flee_from_y = ally.y
                        # If ally is being attacked, flee from what's attacking it
                        if ally.being_attacked_by:
                            for other, odist in enemies:
                                if other.id in ally.being_attacked_by:
                                    flee_from_x = other.x
                                    flee_from_y = other.y
                                    break
                        cell.steer_away(flee_from_x, flee_from_y, 0.85)
                        if cell.sprint_energy > 15 and cell.sprint_cooldown <= 0:
                            cell.sprinting = True
                        # Unique zigzag phase (not synchronized, harder to catch)
                        cell.flee_zigzag_phase += 0.3 + cell.id * 0.01
                        zigzag = math.sin(cell.flee_zigzag_phase) * 0.6
                        cell.desired_angle += zigzag

        # --- Predator eavesdropping: enemy taunts attract hungry predator ---
        if cell.genome.is_predator() and cell.ticks_without_food > 40:
            for prey, dist in enemies:
                if prey.taunt_timer <= 0 or prey.taunt_type is None:
                    continue
                # Wavefront: only when the edge passes
                wave_speed = 3.0 + prey.genome.taunt_power * 3.0
                if not (prey.taunt_radius - wave_speed <= dist <= prey.taunt_radius):
                    continue
                # Prey is making noise -> go there! Hungrier = stronger attraction
                hunger_factor = min(1.0, cell.ticks_without_food / 200.0)
                pull_thrust = 0.3 + hunger_factor * 0.5
                cell.steer_towards(prey.x, prey.y, pull_thrust)
                cell.target_id = prey.id
                cell.decision_target_x = prey.x
                cell.decision_target_y = prey.y
                cell.decision_cooldown = 30
                # Very hungry + close -> sprint
                if cell.ticks_without_food > 120 and dist < sense * 0.5:
                    if cell.sprint_energy > 15 and cell.sprint_cooldown <= 0:
                        cell.sprinting = True
                break  # React to nearest taunt

        # --- Mate-seeking FIRST: if specific mate/scent found, takes over control ---
        mate_handled = False
        if cell.seeking_mate:
            if allies:
                best_mate = None
                best_mate_dist = cell.genome.sense_range * 1.5
                for ally, dist in allies:
                    if ally.wants_to_mate() and dist < best_mate_dist:
                        best_mate = ally
                        best_mate_dist = dist
                        cell.mate_target_id = ally.id
                if best_mate:
                    mate_thrust = 0.7 if best_mate_dist > 40 else 0.9
                    cell.steer_towards(best_mate.x, best_mate.y, mate_thrust)
                    if best_mate_dist < cell.genome.sense_range * 0.5:
                        if cell.sprint_energy > 15 and cell.sprint_cooldown <= 0:
                            cell.sprinting = True
                    mate_handled = True
            if not mate_handled:
                gx, gy = self.pheromones.read_gradient(cell.x, cell.y, PheromoneMap.MATE)
                mate_strength = abs(gx) + abs(gy)
                if mate_strength > 0.1:
                    mate_thrust = min(0.8, 0.3 + mate_strength * 0.3)
                    cell.steer_towards(cell.x + gx * 50, cell.y + gy * 50, mate_thrust)
                    if mate_strength > 0.5 and cell.sprint_energy > 20:
                        cell.sprinting = True
                    cell.mate_target_id = None
                    mate_handled = True
            if not mate_handled:
                # Can't see mate, can't smell scent -> species AI handles movement (patrol/graze)
                # Herbivore/omnivore migrates, predator goes to normal patrol
                cell.mate_target_id = None
                if not cell.genome.is_predator() and not cell.migrating:
                    cell.start_migration()

        # Species-specific AI — ONLY if mate-seeking didn't take over
        if not mate_handled:
            if cell.genome.is_predator():
                self._predator_ai(cell, allies, enemies)
            elif cell.genome.is_omnivore():
                self._omnivore_ai(cell, allies, enemies)
            else:
                self._herbivore_ai(cell, allies, enemies)

        # --- Social attraction: stronger toward leader ---
        if cell.genome.social > 0.15 and allies:
            if best_leader and cell.leader_id:
                # Follow leader (stronger)
                tx = best_leader.x
                ty = best_leader.y
                dx = tx - cell.x
                dy = ty - cell.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > cell.radius * 4:
                    strength = 0.1 * cell.genome.social * min(dist / 80, 2.0)
                    cell.desired_angle = math.atan2(
                        math.sin(cell.desired_angle) + dy / max(dist, 1) * strength,
                        math.cos(cell.desired_angle) + dx / max(dist, 1) * strength
                    )
                # If leader is migrating, follower also migrates
                if best_leader.migrating and not cell.migrating and cell.genome.social > 0.3:
                    cell.migrating = True
                    cell.migrate_x = best_leader.migrate_x + random.gauss(0, 20)
                    cell.migrate_y = best_leader.migrate_y + random.gauss(0, 20)
            else:
                # No leader, pull toward average
                sx = sum(a.x for a, _ in allies) / len(allies)
                sy = sum(a.y for a, _ in allies) / len(allies)
                dx = sx - cell.x
                dy = sy - cell.y
                dist = math.sqrt(dx * dx + dy * dy)

                if dist > cell.radius * 3:
                    strength = 0.06 * cell.genome.social * min(dist / 100, 1.5)
                    cell.desired_angle = math.atan2(
                        math.sin(cell.desired_angle) + dy / max(dist, 1) * strength,
                        math.cos(cell.desired_angle) + dx / max(dist, 1) * strength
                    )
            # Repulsion if too close
            for ally, adist in allies:
                if adist < cell.radius * 2:
                    dx = cell.x - ally.x
                    dy = cell.y - ally.y
                    cell.desired_angle = math.atan2(
                        math.sin(cell.desired_angle) + dy / max(adist, 1) * 0.02,
                        math.cos(cell.desired_angle) + dx / max(adist, 1) * 0.02
                    )
                    break

            # --- Speed synchronization: move at similar speeds ---
            if cell.genome.social > 0.4 and len(allies) >= 2:
                avg_vx = sum(a.vx for a, _ in allies) / len(allies)
                avg_vy = sum(a.vy for a, _ in allies) / len(allies)
                sync = cell.genome.social * 0.1
                cell.vx += (avg_vx - cell.vx) * sync
                cell.vy += (avg_vy - cell.vy) * sync

    def _predator_ai(self, cell, allies, enemies):
        """Predator AI — realistic predator behavior.

        Phases:
        1. RESTING: well-fed -> barely moves, saves energy (lion sleeps 20h/day)
        2. SCAVENGING: hungry -> corpse nearby? eat that, cheap calories
        3. STALKING: prey detected -> slowly, quietly approach (don't scare it off)
        4. AMBUSH STRIKE: close enough -> explosive sprint, short, all-in
        5. PURSUIT: didn't catch -> short chase, ROI-based giving up
        6. REST after hunt: unsuccessful -> stop, recharge sprint
        7. SEARCHING: no prey -> scent/oasis/ambush spot, territorial patrol
        """
        sense = cell.genome.sense_range * self.settings.get('pred_sense', 1.0)
        hunger = cell.ticks_without_food

        # === Post-unsuccessful-hunt rest (SHORT) ===
        rest_ticks = getattr(cell, 'rest_after_hunt', 0)
        if rest_ticks > 0:
            cell.rest_after_hunt = rest_ticks - 1
            cell.thrust = 0.05  # Slow movement, not full stop
            cell.sprinting = False
            # But if prey literally walks onto it -> strike anyway
            for other, dist in enemies:
                if dist < cell.radius * 4 and not other.genome.is_predator():
                    cell.rest_after_hunt = 0  # Wake up!
                    break
            # If very hungry, don't rest further
            if hunger > 150:
                cell.rest_after_hunt = 0
            if cell.rest_after_hunt > 0:
                return

        # Pack mates' target
        pack_target_id = None
        if cell.genome.social > 0.4:
            for ally, _ in allies:
                if ally.target_id is not None:
                    pack_target_id = ally.target_id
                    break
                if ally.hunting_spots and not cell.hunting_spots:
                    cell.hunting_spots = [(x, y, i * 0.5) for x, y, i in ally.hunting_spots[:2]]

        # === PHASE 1: Scavenging FIRST — don't walk away from a fresh carcass! ===
        nearby_meat = self.food_grid.query(cell.x, cell.y, sense)
        best_meat = None
        best_meat_score = -1
        best_meat_dist = sense
        for idx in nearby_meat:
            if idx >= len(self.food):
                continue
            f = self.food[idx]
            if len(f) > 3 and f[3] and f[2] > FOOD_MIN_ENERGY:
                corpse_tick = f[4] if len(f) > 4 else 0
                corpse_age = self.tick - corpse_tick if corpse_tick > 0 else 9999
                # Predator doesn't eat rotten corpse (>2000 ticks old)
                if corpse_age > 2000:
                    continue
                dx = f[0] - cell.x
                dy = f[1] - cell.y
                d = math.sqrt(dx*dx + dy*dy)
                if d < sense:
                    # Fresher + closer = better
                    freshness = max(0, 1.0 - corpse_age / 2000.0)
                    score = freshness * 2.0 + (sense - d) / sense
                    if score > best_meat_score:
                        best_meat = (f[0], f[1])
                        best_meat_score = score
                        best_meat_dist = d

        if best_meat:
            # Hungry predator: ALWAYS chooses corpse (zero risk)
            eat_meat = True
            # Only very aggressive + very close live prey -> hunts instead
            if cell.genome.aggression > 0.8 and enemies:
                closest_prey_dist = min((d for _, d in enemies), default=sense)
                if closest_prey_dist < best_meat_dist * 0.3:
                    eat_meat = False
            if eat_meat:
                # Close -> STOP to eat (thrust 0), farther -> walk over
                eat_dist = cell.radius + FOOD_RADIUS + 5  # Eating range + small margin
                thrust = 0.4 if best_meat_dist > eat_dist else 0.0
                cell.steer_towards(best_meat[0], best_meat[1], thrust)
                cell.target_id = None
                cell.sprinting = False
                return

        # === PHASE 2: Well-fed rest (ONLY if no corpse nearby) ===
        if hunger < 30 and not cell.ambushing:
            cell.thrust = 0.08 + random.random() * 0.05
            cell.target_id = None
            cell.sprinting = False
            if random.random() < 0.06:
                cell.desired_angle += random.gauss(0, 0.4)
            return

        # === PHASES 3-5: Prey search, stalking, ambush strike ===
        best_prey = None
        best_score = -1

        all_nearby = self.cell_grid.query(cell.x, cell.y, sense)
        for other in all_nearby:
            if other.id == cell.id or not other.alive:
                continue
            if other.genome.is_predator():
                continue
            if other.hibernating:
                continue

            dx = other.x - cell.x
            dy = other.y - cell.y
            dist = max(math.sqrt(dx * dx + dy * dy), 1)

            my_damage = cell.damage_per_tick
            ticks_to_kill = other.hp / max(0.1, my_damage)
            pack_attackers = 1 + sum(1 for a, _ in allies if a.target_id == other.id)
            ticks_with_pack = ticks_to_kill / pack_attackers

            # Too large prey alone
            if ticks_to_kill > 25 and pack_attackers < 2:
                continue

            my_speed = cell.genome.max_speed * cell.effective_speed_mult
            prey_speed = other.genome.max_speed * other.effective_speed_mult

            # Faster prey + far + no sprint -> hopeless
            if prey_speed > my_speed * 1.1 and dist > sense * 0.25:
                if cell.sprint_energy < 30 or cell.sprint_cooldown > 0:
                    continue

            # === Scoring: realistic predator logic ===
            score = 0.0

            # Proximity (most important — energy efficiency)
            score += (sense - dist) / sense * 1.5

            # Size difference (smaller is easier to catch AND kill)
            score += (cell.current_size - other.current_size) * 0.04

            # Speed advantage (slower = more certain prey)
            speed_ratio = my_speed / max(prey_speed, 0.1)
            score += min(speed_ratio - 1.0, 0.5) * 0.8  # Max +0.4

            # Injured prey (in nature, predators target the sick/injured)
            score += (1.0 - other.hp_ratio) * 0.8

            # Juvenile/young (easy target, like in nature)
            if other.is_child:
                score += 0.5

            # ISOLATED prey (cut off from herd — the most important strategy!)
            if other.pack_mates < 2:
                score += 0.6  # Solitary = ideal target
            elif other.pack_mates > 5:
                score -= 0.4  # Large herd = confusion effect + defense

            # Pack target bonus
            if pack_target_id and other.id == pack_target_id:
                score += 0.6 * cell.genome.social

            # Kill speed
            score -= ticks_with_pack * 0.015

            # Learned preferences (from memory)
            if cell.prefer_weak > 0:
                score += (5.0 - other.genome.defense) * cell.prefer_weak * 0.08
            if cell.prefer_isolated > 0:
                score += (1.0 if other.pack_mates < 2 else -0.3) * cell.prefer_isolated * 0.2
            if cell.prefer_slow > 0:
                score += (2.0 - other.genome.max_speed) * cell.prefer_slow * 0.1

            # Hunger: less picky
            if hunger > 200:
                score += 0.4

            # Target loyalty: don't switch every frame!
            if cell.chase_target_id == other.id:
                score += 3.0

            if score > best_score:
                best_prey = other
                best_score = score

        if best_prey:
            dx = best_prey.x - cell.x
            dy = best_prey.y - cell.y
            prey_dist = math.sqrt(dx * dx + dy * dy)

            # Attack decision
            will_attack = (cell.ambushing or
                           hunger > 200 or
                           prey_dist < sense * 0.3 or
                           random.random() < cell.genome.aggression * 1.3)

            if not will_attack:
                cell.wander()
                return

            # === ROI hunting ===
            expected_corpse = best_prey.energy * 0.5 + best_prey.genome.size ** 2 * 0.8
            roi_limit = expected_corpse * 0.35

            if cell.chase_target_id == best_prey.id:
                cell.chase_energy_spent += cell.energy_cost_per_tick() * (2.5 if cell.sprinting else 1.0)
                if cell.chase_energy_spent > roi_limit:
                    # Give up + rest (real predator doesn't keep running)
                    cell.target_id = None
                    cell.sprinting = False
                    cell.chase_energy_spent = 0.0
                    cell.chase_target_id = None
                    cell.rest_after_hunt = 20 + random.randint(0, 20)  # Short rest
                    cell.remember_hunt(best_prey, False, best_prey.x, best_prey.y)
                    return
            else:
                cell.chase_target_id = best_prey.id
                cell.chase_energy_spent = 0.0

            # === Ambush attack — instant sprint ===
            if cell.ambushing:
                cell.ambushing = False
                cell.hiding = False
                cell.ambush_ticks = 0
                cell.sprinting = True
                cell.sprint_energy = max(cell.sprint_energy, 70)

            cell.target_id = best_prey.id

            # === STALKING vs AMBUSH STRIKE phase ===
            strike_range = sense * 0.3  # Within this: full charge

            if prey_dist > strike_range:
                # --- STALKING: slow, quiet approach ---
                # Doesn't scare off prey. Sprint OFF. Low thrust.
                cell.sprinting = False
                # Interceptor: get ahead of it, not behind
                prey_vel = math.sqrt(best_prey.vx ** 2 + best_prey.vy ** 2)
                if prey_vel > 0.3:
                    intercept_time = min(prey_dist / max(prey_vel + 1, 1), 50)
                    target_x = best_prey.x + best_prey.vx * intercept_time * 0.15
                    target_y = best_prey.y + best_prey.vy * intercept_time * 0.15
                else:
                    target_x = best_prey.x
                    target_y = best_prey.y

                # Approach speed: slow! Stealth gene -> even slower (but more effective)
                stalk_thrust = 0.2 + (hunger / 1000) * 0.15  # Hungrier -> slightly faster
                stalk_thrust *= (1.0 - cell.genome.stealth * 0.35)  # Stealth gene slows down
                stalk_thrust = min(stalk_thrust, 0.4)
                cell.steer_towards(target_x, target_y, stalk_thrust)
            else:
                # --- AMBUSH STRIKE: explosive sprint! ---
                if cell.sprint_energy > 15 and cell.sprint_cooldown <= 0:
                    cell.sprinting = True

                # Interceptor up close too
                prey_vel = math.sqrt(best_prey.vx ** 2 + best_prey.vy ** 2)
                if prey_vel > 0.5 and prey_dist > cell.radius * 2:
                    my_speed = max(1, math.sqrt(cell.vx ** 2 + cell.vy ** 2))
                    t = min(prey_dist / max(my_speed + prey_vel, 1), 20)
                    target_x = best_prey.x + best_prey.vx * t * 0.25
                    target_y = best_prey.y + best_prey.vy * t * 0.25
                else:
                    target_x = best_prey.x
                    target_y = best_prey.y

                cell.steer_towards(target_x, target_y, 1.0)  # FULL THROTTLE

            # === Pack hunting signal ===
            if cell.genome.social > 0.4:
                for ally, adist in allies:
                    if adist < sense * 0.6 and ally.genome.social > 0.3:
                        if ally.target_id is None or ally.target_id != best_prey.id:
                            ally.target_id = best_prey.id

            # === Encirclement: approach from the side (ONLY if not in bite range!) ===
            if cell.genome.social > 0.4 and allies:
                pack_count = sum(1 for a, d in allies if a.target_id == best_prey.id)
                # Within bite range -> DON'T encircle, BITE!
                if pack_count > 0 and (cell.radius * 3 < prey_dist < strike_range * 2):
                    d = max(prey_dist, 1)
                    side = 1 if cell.id % 2 == 0 else -1
                    offset = cell.radius * 4 * side
                    tx = best_prey.x + (-dy / d) * offset
                    ty = best_prey.y + (dx / d) * offset
                    cell.steer_towards(tx, ty, 0.8)
        else:
            # === PHASE 7: No prey — territorial patrol ===
            cell.target_id = None
            cell.sprinting = False
            cell.chase_energy_spent = 0.0
            cell.chase_target_id = None

            # Starvation-based wandering
            is_starving = hunger > 250

            if is_starving and not cell.migrating:
                cell.migrating = True
                spot = cell.best_hunting_spot()
                if spot:
                    cell.migrate_x = spot[0] + random.gauss(0, 50)
                    cell.migrate_y = spot[1] + random.gauss(0, 50)
                else:
                    # Oasis search — herbivores will be there
                    best_oasis = None
                    best_oasis_score = -1
                    for src in self.food_sources:
                        dx = src['x'] - cell.x
                        dy = src['y'] - cell.y
                        d = math.sqrt(dx * dx + dy * dy)
                        oasis_score = src['richness'] * 10 - d * 0.01
                        if oasis_score > best_oasis_score:
                            best_oasis = src
                            best_oasis_score = oasis_score
                    if best_oasis:
                        cell.migrate_x = best_oasis['x'] + random.gauss(0, 80)
                        cell.migrate_y = best_oasis['y'] + random.gauss(0, 80)
                    else:
                        angle = random.uniform(0, 2 * math.pi)
                        cell.migrate_x = cell.x + math.cos(angle) * 500
                        cell.migrate_y = cell.y + math.sin(angle) * 500
                cell.migrate_x = max(50, min(self.width - 50, cell.migrate_x))
                cell.migrate_y = max(50, min(self.height - 50, cell.migrate_y))
                if cell.leader_id is None and cell.genome.social > 0.3:
                    for ally, adist in allies:
                        if ally.leader_id == cell.id and not ally.migrating:
                            ally.migrating = True
                            ally.migrate_x = cell.migrate_x + random.gauss(0, 30)
                            ally.migrate_y = cell.migrate_y + random.gauss(0, 30)

            if cell.migrating:
                dx = cell.migrate_x - cell.x
                dy = cell.migrate_y - cell.y
                dist_to_target = math.sqrt(dx * dx + dy * dy)
                if dist_to_target < 60:
                    cell.migrating = False
                    if hunger > 200:
                        cell.start_migration()
                else:
                    prey_smell = self.pheromones.read(cell.x, cell.y, PheromoneMap.PREY_SCENT, radius=2)
                    if prey_smell > 0.8:
                        cell.migrating = False  # Prey scent -> stop to hunt
                    else:
                        cell.steer_towards(cell.migrate_x, cell.migrate_y, 0.45)
                        return

            # --- Ambushing near oasis ---
            if cell.in_shelter and cell.energy > 20:
                cell.ambushing = True
                cell.ambush_ticks += 1
                cell.thrust = 0.0
                cell.hiding = True
                self.pheromones.deposit(cell.x, cell.y, PheromoneMap.AMBUSH, 0.1)
                if cell.ambush_ticks > 600:
                    cell.ambushing = False
                    cell.ambush_ticks = 0
                return
            elif not cell.in_shelter:
                cell.ambushing = False
                cell.ambush_ticks = 0
                cell.hiding = False

            # === Decision loyalty: if active target, hold direction ===
            if cell.decision_cooldown > 0:
                cell.decision_cooldown -= 1
                dx = cell.decision_target_x - cell.x
                dy = cell.decision_target_y - cell.y
                dist_to_dec = math.sqrt(dx * dx + dy * dy)
                if dist_to_dec > 20:
                    # Also watches en route: if prey scent detected, stops and hunts
                    prey_smell = self.pheromones.read(cell.x, cell.y, PheromoneMap.PREY_SCENT, radius=1)
                    if prey_smell > 1.0:
                        cell.decision_cooldown = 0  # Scent -> switch to hunt mode
                    else:
                        cell.steer_towards(cell.decision_target_x, cell.decision_target_y, 0.3)
                        return
                else:
                    cell.decision_cooldown = 0  # Arrived

            # --- Corpse scent tracking (highest priority during patrol!) ---
            cgx, cgy = self.pheromones.read_gradient(cell.x, cell.y, PheromoneMap.CORPSE_SCENT)
            corpse_smell = abs(cgx) + abs(cgy)
            if corpse_smell > 0.1:
                # Corpse scent -> go there quickly, free calories!
                corpse_thrust = min(0.6, 0.3 + corpse_smell * 0.2)
                target_x = cell.x + cgx * 60
                target_y = cell.y + cgy * 60
                cell.decision_target_x = target_x
                cell.decision_target_y = target_y
                cell.decision_cooldown = 20
                cell.steer_towards(target_x, target_y, corpse_thrust)
                return

            # --- Scent tracking (with decision loyalty) ---
            gx, gy = self.pheromones.read_gradient(cell.x, cell.y, PheromoneMap.PREY_SCENT)
            scent_strength = abs(gx) + abs(gy)
            if scent_strength > 0.12:
                scent_thrust = 0.25 if hunger < 150 else 0.4
                # Lock target: hold direction for 30 ticks
                target_x = cell.x + gx * 50
                target_y = cell.y + gy * 50
                cell.decision_target_x = target_x
                cell.decision_target_y = target_y
                cell.decision_cooldown = 25
                cell.steer_towards(target_x, target_y, scent_thrust)
                return

            gx2, gy2 = self.pheromones.read_gradient(cell.x, cell.y, PheromoneMap.TRAIL)
            if abs(gx2) + abs(gy2) > 0.15:
                target_x = cell.x + gx2 * 40
                target_y = cell.y + gy2 * 40
                cell.decision_target_x = target_x
                cell.decision_target_y = target_y
                cell.decision_cooldown = 20
                cell.steer_towards(target_x, target_y, 0.2)
                return

            # --- Proactive: toward oasis (herbivores are there) ---
            if hunger > 80:
                best_oasis = None
                best_oasis_dist = 9999
                for src in self.food_sources:
                    dx = src['x'] - cell.x
                    dy = src['y'] - cell.y
                    d = math.sqrt(dx * dx + dy * dy)
                    if d < best_oasis_dist and d > 40:
                        best_oasis = src
                        best_oasis_dist = d
                if best_oasis and best_oasis_dist < sense * 5:
                    cell.decision_target_x = best_oasis['x']
                    cell.decision_target_y = best_oasis['y']
                    cell.decision_cooldown = 40
                    cell.steer_towards(best_oasis['x'], best_oasis['y'], 0.25)
                    return

            # --- Ambush spot search ---
            best_ambush = None
            best_ambush_dist = 9999
            for s in self.shelters:
                prey_near = self.pheromones.read(s['x'], s['y'], PheromoneMap.PREY_SCENT)
                food_near = self.pheromones.read(s['x'], s['y'], PheromoneMap.FOOD_HERE)
                if prey_near > 0.2 or food_near > 0.3:
                    dx = s['x'] - cell.x
                    dy = s['y'] - cell.y
                    d = math.sqrt(dx*dx + dy*dy)
                    if d < best_ambush_dist:
                        best_ambush = s
                        best_ambush_dist = d

            if best_ambush and best_ambush_dist > 15:
                cell.decision_target_x = best_ambush['x']
                cell.decision_target_y = best_ambush['y']
                cell.decision_cooldown = 30
                cell.steer_towards(best_ambush['x'], best_ambush['y'], 0.2)
                return

            # Known hunting ground
            spot = cell.best_hunting_spot()
            if spot:
                dx = spot[0] - cell.x
                dy = spot[1] - cell.y
                if math.sqrt(dx*dx + dy*dy) > 30:
                    cell.decision_target_x = spot[0]
                    cell.decision_target_y = spot[1]
                    cell.decision_cooldown = 35
                    cell.steer_towards(spot[0], spot[1], 0.3)
                    return

            # Active territorial patrol — ALWAYS goes toward oasis, doesn't stand still!
            # Search for nearest oasis (from anywhere)
            best_oasis = None
            best_oasis_dist = 9999
            for src in self.food_sources:
                dx = src['x'] - cell.x
                dy = src['y'] - cell.y
                d = math.sqrt(dx * dx + dy * dy)
                if d < best_oasis_dist and d > 30:
                    best_oasis = src
                    best_oasis_dist = d
            if best_oasis:
                cell.steer_towards(best_oasis['x'], best_oasis['y'], 0.25)
                cell.decision_target_x = best_oasis['x']
                cell.decision_target_y = best_oasis['y']
                cell.decision_cooldown = 30
            else:
                # No oasis -> random direction change
                cell.thrust = 0.2
                cell.desired_angle += random.gauss(0, 0.5)

    def _omnivore_ai(self, cell, allies, enemies):
        """Omnivore AI: eats plants and carrion, but doesn't hunt."""
        sense = cell.genome.sense_range

        # --- Under active attack: immediate flee ---
        if cell.being_attacked_by:
            for other, dist in enemies:
                if other.id in cell.being_attacked_by:
                    cell.steer_away(other.x, other.y, 1.0)
                    if cell.sprint_energy > 10 and cell.sprint_cooldown <= 0:
                        cell.sprinting = True
                    return
            cell.desired_angle += math.pi
            cell.thrust = 0.9
            return

        # --- Danger detection: flee from predators (stealth system) ---
        danger = None
        danger_dist = sense + 1
        detect_threshold = max(0.15, 0.4 - cell.genome.defense * 0.03)
        for other, dist in enemies:
            if other.genome.is_predator() and other.genome.attack > cell.genome.defense * 0.3:
                visibility = self.calc_detection(cell, other, dist)
                if visibility < detect_threshold:
                    continue  # Doesn't detect stealthy predator
                if dist < danger_dist:
                    danger = other
                    danger_dist = dist

        if danger and danger_dist < sense * 0.8:
            cell.steer_away(danger.x, danger.y, 0.9)
            if danger_dist < sense * 0.35 and cell.sprint_energy > 15 and cell.sprint_cooldown <= 0:
                cell.sprinting = True
            # Zigzag flee (like herbivore)
            if danger_dist < sense * 0.25 and cell.genome.maneuverability > 0.3:
                zigzag = (1 if cell.age % 8 < 4 else -1) * cell.genome.maneuverability * 0.4
                cell.desired_angle += zigzag
            return

        # --- Pheromone sensing: danger scent ---
        danger_smell = self.pheromones.read(cell.x, cell.y, PheromoneMap.DANGER, radius=2)
        if danger_smell > 1.5:
            gx, gy = self.pheromones.read_gradient(cell.x, cell.y, PheromoneMap.DANGER)
            if abs(gx) + abs(gy) > 0.2:
                cell.steer_away(cell.x + gx * 30, cell.y + gy * 30, 0.6)
                return

        # --- Carrion search: OLD corpse nearby? (fresh corpse belongs to predator!) ---
        food_sense = sense * 1.4  # Good sense for carrion
        nearby_food = self.food_grid.query(cell.x, cell.y, food_sense)
        best_meat = None
        best_meat_dist = food_sense
        best_plant = None
        best_plant_dist = food_sense

        for idx in nearby_food:
            if idx >= len(self.food):
                continue
            f = self.food[idx]
            fx, fy, fe = f[0], f[1], f[2]
            is_meat = f[3] if len(f) > 3 else False
            dx = fx - cell.x
            dy = fy - cell.y
            d = math.sqrt(dx * dx + dy * dy)
            if d > food_sense:
                continue
            if is_meat:
                # Omnivore: only old corpse (>150 ticks) — fresh is dangerous (predator still there!)
                corpse_tick = f[4] if len(f) > 4 else 0
                corpse_age = self.tick - corpse_tick if corpse_tick > 0 else 9999
                if corpse_age > 800 and d < best_meat_dist:
                    best_meat = (fx, fy, fe)
                    best_meat_dist = d
            elif not is_meat and d < best_plant_dist:
                best_plant = (fx, fy, fe)
                best_plant_dist = d

        # Carrion preference: old corpse (safe)
        eat_dist = cell.radius + FOOD_RADIUS + 5
        if best_meat and best_meat_dist < food_sense:
            thrust = 0.5 if best_meat_dist > eat_dist else 0.15
            cell.steer_towards(best_meat[0], best_meat[1], thrust)
            return

        # Plant search
        if best_plant and best_plant_dist < food_sense:
            thrust = 0.4 if best_plant_dist > eat_dist else 0.1
            cell.steer_towards(best_plant[0], best_plant[1], thrust)
            return

        # --- Corpse scent tracking (old corpses!) ---
        cgx, cgy = self.pheromones.read_gradient(cell.x, cell.y, PheromoneMap.CORPSE_SCENT)
        corpse_smell = abs(cgx) + abs(cgy)
        if corpse_smell > 0.15:
            cell.steer_towards(cell.x + cgx * 40, cell.y + cgy * 40, 0.4)
            return

        # --- Pheromone: food scent ---
        gx, gy = self.pheromones.read_gradient(cell.x, cell.y, PheromoneMap.FOOD_HERE)
        if abs(gx) + abs(gy) > 0.15:
            cell.steer_towards(cell.x + gx * 40, cell.y + gy * 40, 0.35)
            return

        # --- Migration if hungry ---
        if cell.ticks_without_food > 120 and not cell.migrating:
            cell.start_migration()
            return

        if cell.migrating:
            dx = cell.migrate_x - cell.x
            dy = cell.migrate_y - cell.y
            dist_to_target = math.sqrt(dx * dx + dy * dy)
            if dist_to_target < 50:
                cell.migrating = False
                if cell.ticks_without_food > 80:
                    cell.start_migration()
            else:
                cell.steer_towards(cell.migrate_x, cell.migrate_y, 0.5)
                if cell.ticks_without_food < 20:
                    cell.migrating = False
            return

        cell.wander()

    def _herbivore_ai(self, cell, allies, enemies):
        """Herbivore AI — realistic prey behavior.

        Flee phases:
        1. UNDER ACTIVE ATTACK: being damaged -> immediate panic flee + sprint
        2. PREDATOR NEARBY: visual detection -> zigzag flee or freeze
        3. DANGER SCENT: pheromone detection -> shelter search or avoidance
        4. HERD PROTECTION: if large herd -> braver, doesn't flee immediately
        """
        sense = cell.genome.sense_range

        # === 1. UNDER ACTIVE ATTACK — immediate panic reaction ===
        # If being damaged -> IMMEDIATELY flee, highest priority!
        if cell.being_attacked_by:
            # Find the attacker
            attacker = None
            for other, dist in enemies:
                if other.id in cell.being_attacked_by:
                    attacker = other
                    break
            if attacker:
                # Panic flee + sprint + zigzag
                cell.steer_away(attacker.x, attacker.y, 1.0)
                if cell.sprint_energy > 10 and cell.sprint_cooldown <= 0:
                    cell.sprinting = True
                # Zigzag: maneuverable cells also step sideways
                if cell.genome.maneuverability > 0.3:
                    zigzag = (1 if cell.age % 8 < 4 else -1) * cell.genome.maneuverability * 0.5
                    cell.desired_angle += zigzag
                # Danger alert to the herd!
                if cell.genome.social > 0.2:
                    for ally, dist in allies:
                        if dist < sense * 0.7 and ally.genome.social > 0.15:
                            ally.alert = True
                            ally.alert_x = attacker.x
                            ally.alert_y = attacker.y
                cell.remember_danger(attacker.x, attacker.y)
                return
            else:
                # Attacker not visible -> flee opposite to facing direction
                cell.desired_angle += math.pi  # 180 degree turn
                cell.thrust = 1.0
                if cell.sprint_energy > 10 and cell.sprint_cooldown <= 0:
                    cell.sprinting = True
                return

        # --- Camouflage: hide in shelter when danger is present ---
        if cell.in_shelter and cell.hiding:
            cell.thrust = 0.05
            if not enemies:
                danger_smell = self.pheromones.read(cell.x, cell.y, PheromoneMap.DANGER)
                if danger_smell < 0.3:
                    cell.hiding = False

        # === 2. PREDATOR NEARBY — visual detection (stealth system) ===
        danger = None
        danger_dist = sense + 1
        # Detection threshold: prey's sensitivity (high defense -> more alert)
        detect_threshold = max(0.15, 0.4 - cell.genome.defense * 0.03)

        for other, dist in enemies:
            if other.genome.is_predator() and other.genome.attack > cell.genome.defense * 0.3:
                # Stealth-based detection: slow predator invisible from afar
                visibility = self.calc_detection(cell, other, dist)
                if visibility < detect_threshold:
                    continue  # Doesn't notice — too stealthy / too far
                if dist < danger_dist:
                    danger = other
                    danger_dist = dist

        # Danger alert from allies
        if not danger and cell.alert and cell.genome.social > 0.3:
            cell.steer_away(cell.alert_x, cell.alert_y, 0.7)
            cell.alert = False
            if cell.sprint_energy > 30 and cell.sprint_cooldown <= 0:
                cell.sprinting = True  # Warning from allies -> sprint
            return

        if danger and danger_dist < sense * 0.8:
            # Flee strategy based on distance:
            if danger_dist < sense * 0.2:
                # VERY CLOSE — panic sprint + zigzag
                cell.steer_away(danger.x, danger.y, 1.0)
                if cell.sprint_energy > 10 and cell.sprint_cooldown <= 0:
                    cell.sprinting = True
                # Zigzag maneuver
                if cell.genome.maneuverability > 0.3:
                    zigzag = (1 if cell.age % 6 < 3 else -1) * cell.genome.maneuverability * 0.6
                    cell.desired_angle += zigzag
            elif danger_dist < sense * 0.4:
                # CLOSE — fast flee, shelter search
                cell.steer_away(danger.x, danger.y, 0.9)
                if cell.sprint_energy > 25 and cell.sprint_cooldown <= 0:
                    cell.sprinting = True
                # Nearby shelter?
                for s in self.shelters:
                    dx = s['x'] - cell.x
                    dy = s['y'] - cell.y
                    d = math.sqrt(dx*dx + dy*dy)
                    # Shelter only if NOT in predator's direction
                    if d < sense * 0.4:
                        shelter_angle = math.atan2(dy, dx)
                        danger_angle = math.atan2(danger.y - cell.y, danger.x - cell.x)
                        angle_diff = abs(math.atan2(math.sin(shelter_angle - danger_angle),
                                                     math.cos(shelter_angle - danger_angle)))
                        if angle_diff > 1.5:  # Shelter opposite to predator
                            cell.steer_towards(s['x'], s['y'], 0.85)
                            cell.hiding = True
                            break
            else:
                # FARTHER — alert flee, no sprint (saves energy)
                cell.steer_away(danger.x, danger.y, 0.65)

            cell.remember_danger(danger.x, danger.y)

            # --- Danger alert to the herd ---
            if cell.genome.social > 0.3:
                for ally, dist in allies:
                    if dist < sense * 0.7 and ally.genome.social > 0.15:
                        ally.alert = True
                        ally.alert_x = danger.x
                        ally.alert_y = danger.y
                        ally.remember_danger(danger.x, danger.y)

            # --- Herd protection: large herd -> braver ---
            if cell.pack_mates >= 4 and cell.genome.social > 0.5:
                if danger_dist > sense * 0.35:
                    # Enough herd and far enough -> don't flee, keep eating
                    self._seek_food_or_wander(cell)
                    return

            # --- Migration decision: harassed too much here? ---
            if cell.should_migrate() and not cell.migrating:
                cell.start_migration()
                if cell.genome.social > 0.4:
                    for ally, dist in allies:
                        if ally.genome.social > 0.3 and not ally.migrating:
                            ally.migrate_x = cell.migrate_x + random.gauss(0, 30)
                            ally.migrate_y = cell.migrate_y + random.gauss(0, 30)
                            ally.migrating = True
                            ally.attacks_here = 0
        else:
            cell.alert = False

            # --- Migration mode: purposefully heads to new pasture ---
            if cell.migrating:
                dx = cell.migrate_x - cell.x
                dy = cell.migrate_y - cell.y
                dist_to_target = math.sqrt(dx * dx + dy * dy)
                if dist_to_target < 50:
                    cell.migrating = False
                    cell.attacks_here = 0
                    # Arrived: food nearby? If not, start new migration
                    if cell.ticks_without_food > 100:
                        cell.start_migration()  # Keep searching
                else:
                    # Also watches en route: if sees food, stops
                    if cell.ticks_without_food < 20:
                        cell.migrating = False  # Found food en route!
                    else:
                        cell.steer_towards(cell.migrate_x, cell.migrate_y, 0.7)
                        cell.sprinting = dist_to_target > 200 and cell.sprint_energy > 40
                        return

            # --- Pasture sharing + danger zone sharing within herd ---
            if cell.genome.social > 0.3 and allies:
                for ally, dist in allies:
                    if ally.genome.social > 0.2:
                        # Pasture sharing
                        if ally.food_spots and not cell.food_spots:
                            cell.food_spots = [(x, y, a * 0.5) for x, y, a in ally.food_spots[:2]]
                        # Danger zone sharing
                        if ally.danger_zones and not cell.danger_zones:
                            cell.danger_zones = [(x, y, d * 0.4) for x, y, d in ally.danger_zones[:2]]
                        break

            # --- Proactively flee from danger zone ---
            current_danger = cell.is_danger_zone(cell.x, cell.y)
            if current_danger > 1.5 and not cell.migrating:
                # This place should be avoided — migrate
                cell.start_migration()
                return

            # Well-fed + safe = energy saving mode
            energy_ratio = cell.energy / max(1, cell.genome.repro_thresh)
            if energy_ratio > 0.9 and not enemies:
                cell.thrust = min(cell.thrust, 0.15)  # Restful drifting

            self._seek_food_or_wander(cell)

    def _seek_food_or_wander(self, cell):
        """Food search or wandering. Predator seeks meat, herbivore seeks plants."""
        sense = cell.genome.sense_range
        is_pred = cell.genome.is_predator()
        # Herbivores detect food better (smell bonus)
        food_sense = sense if is_pred else sense * 1.6
        nearby_food = self.food_grid.query(cell.x, cell.y, food_sense)
        best_food = None
        best_dist = food_sense + 1

        for idx in nearby_food:
            if idx >= len(self.food):
                continue
            food_item = self.food[idx]
            fx, fy = food_item[0], food_item[1]
            is_meat = food_item[3] if len(food_item) > 3 else False
            # Only searches for matching food type
            if is_pred and not is_meat:
                continue
            if not is_pred and is_meat:
                continue
            dx = fx - cell.x
            dy = fy - cell.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < best_dist:
                best_food = (fx, fy)
                best_dist = dist

        if best_food:
            # Energy-efficient food search: distance + hunger based speed
            energy_ratio = cell.energy / max(1, cell.genome.repro_thresh)
            if best_dist < cell.radius * 3:
                food_thrust = 0.2   # Very close: slow approach
            elif best_dist < 40:
                food_thrust = 0.35  # Close: slow
            elif energy_ratio > 0.8:
                food_thrust = 0.3   # Well-fed: relaxed pace
            elif energy_ratio < 0.3:
                food_thrust = 0.75  # Hungry: hurried
            else:
                food_thrust = 0.45  # Normal: medium tempo
            cell.steer_towards(best_food[0], best_food[1], food_thrust)
            return True
        else:
            # No food nearby: search pasture from memory
            spot = cell.best_food_spot()
            if spot and not is_pred:
                # Toward pasture: faster if hungry, slower if well-fed
                energy_ratio = cell.energy / max(1, cell.genome.repro_thresh)
                mem_thrust = 0.3 if energy_ratio > 0.5 else 0.55
                cell.steer_towards(spot[0], spot[1], mem_thrust)
                # If arrived but no food, fade the memory
                dx = spot[0] - cell.x
                dy = spot[1] - cell.y
                if dx * dx + dy * dy < 40 * 40:
                    cell.food_spots = [(x, y, a * 0.85) for x, y, a in cell.food_spots]
                    cell.food_spots = [(x, y, a) for x, y, a in cell.food_spots if a > 0.2]
                return True
            # Herbivore: pheromone-based food search before wandering
            if not is_pred:
                gx, gy = self.pheromones.read_gradient(cell.x, cell.y, PheromoneMap.FOOD_HERE)
                if abs(gx) + abs(gy) > 0.15:
                    target_x = cell.x + gx * 40
                    target_y = cell.y + gy * 40
                    cell.steer_towards(target_x, target_y, 0.4)
                    return True
            # Long time without eating + would wander -> migrate instead!
            if cell.ticks_without_food > 150 and not cell.migrating:
                cell.start_migration()
                return True
            cell.wander()
            return False

    def _eat_food(self, cell):
        """Cell eats nearby food — SLOW eating, food gradually depletes."""
        # Fleeing cell does NOT eat! (doesn't stop to eat while fleeing predator)
        if cell.being_attacked_by:
            return
        if cell.sprinting and cell.thrust > 0.7:
            return  # Sprint+high thrust = active fleeing
        # Non-predator: if predator is very close, DON'T eat!
        if not cell.genome.is_predator() and self.stats["predators"] > 0:
            threat_range = cell.genome.sense_range * 0.35
            threat_range_sq = threat_range * threat_range
            nearby_threats = self.cell_grid.query(cell.x, cell.y, threat_range)
            for other in nearby_threats:
                if other.id != cell.id and other.alive and other.genome.is_predator():
                    dx = other.x - cell.x
                    dy = other.y - cell.y
                    if dx * dx + dy * dy < threat_range_sq:
                        return  # Predator too close — DON'T eat, flee!
        eat_range = cell.radius + FOOD_RADIUS + 2
        eat_range_sq = eat_range * eat_range
        nearby_idxs = self.food_grid.query(cell.x, cell.y, eat_range + SPATIAL_GRID_SIZE)
        is_pred = cell.genome.is_predator()
        is_omni = cell.genome.is_omnivore()

        for idx in nearby_idxs:
            if idx >= len(self.food):
                continue
            food_item = self.food[idx]
            fx, fy, fe = food_item[0], food_item[1], food_item[2]
            # Already eaten this tick (energy set to 0)
            if fe <= FOOD_MIN_ENERGY:
                continue
            is_meat = food_item[3] if len(food_item) > 3 else False
            corpse_tick = food_item[4] if len(food_item) > 4 else 0
            corpse_age = self.tick - corpse_tick if corpse_tick > 0 else 9999
            # Predator: only meat. Herbivore: only plants. Omnivore: everything!
            if is_pred and not is_meat:
                continue
            if not is_pred and not is_omni and is_meat:
                continue
            # Freshness preference: predator prefers fresh, omnivore prefers old
            if is_meat and is_omni and corpse_age < 800:
                continue  # Omnivore won't touch fresh corpse (predator still there)
            if is_meat and is_pred and corpse_age > 2000:
                continue  # Predator won't eat old rotten corpse
            dx = fx - cell.x
            dy = fy - cell.y
            if dx * dx + dy * dy < eat_range_sq:
                # Eating speed — predators eat meat faster
                eat_speed = self.settings['eat_speed']
                if is_pred and is_meat:
                    eat_speed *= 1.8  # Predator: fast meat eating
                bite = min(eat_speed, fe)
                # Digestion efficiency by type
                if is_omni:
                    digest_eff = 0.75  # Omnivore: medium for everything
                elif is_pred:
                    digest_eff = 0.9   # Predator: specialized for meat
                else:
                    digest_eff = 0.9   # Herbivore: specialized for plants
                cell.energy += bite * digest_eff
                new_energy = fe - bite

                # Zero energy in place (DON'T delete from list! cleanup at end of tick)
                if new_energy <= FOOD_MIN_ENERGY:
                    self.stats["total_eaten"] += 1
                    if is_meat and corpse_tick > 0:
                        self.food[idx] = (fx, fy, 0.0, is_meat, corpse_tick)
                    else:
                        self.food[idx] = (fx, fy, 0.0, is_meat)
                else:
                    if is_meat and corpse_tick > 0:
                        self.food[idx] = (fx, fy, new_energy, is_meat, corpse_tick)
                    else:
                        self.food[idx] = (fx, fy, new_energy, is_meat)

                cell.ticks_without_food = 0
                cell.last_eat_tick = self.tick
                cell.hibernating = False  # Eating wakes up
                # Stops while eating
                cell.thrust = min(cell.thrust, 0.1)
                if not is_pred:
                    cell.remember_food(fx, fy)
                    self.pheromones.deposit(fx, fy, PheromoneMap.FOOD_HERE, 0.2)
                else:
                    self.pheromones.deposit(fx, fy, PheromoneMap.FOOD_HERE, 0.3)
                break  # Eats only 1 food at a time

    def _predator_attack(self, predator):
        """Predator damages prey — per-tick combat, not instant kill."""
        attack_range = predator.radius + 8
        nearby = self.cell_grid.query(predator.x, predator.y, attack_range)

        for prey in nearby:
            if prey.id == predator.id or not prey.alive:
                continue
            if prey.genome.is_predator():
                continue
            dx = prey.x - predator.x
            dy = prey.y - predator.y
            dist_sq = dx * dx + dy * dy
            strike_dist = predator.radius + prey.radius + 3
            if dist_sq < strike_dist * strike_dist:
                dist = math.sqrt(dist_sq)  # Only sqrt when actually fighting (for knockback)
                # --- Confusion effect: swarm confusion ---
                # 5+ small herbivores together -> predator gets confused
                if prey.pack_mates >= 5:
                    small_count = 0
                    nearby_herbs = self.cell_grid.query(prey.x, prey.y, 40)
                    for h in nearby_herbs:
                        if h.alive and not h.genome.is_predator() and h.current_size < 10:
                            small_count += 1
                    if small_count >= 5 and random.random() < 0.4:
                        # Swarm confusion: misses the attack
                        predator.energy -= 0.5
                        continue

                # --- Damage calculation ---
                damage = predator.damage_per_tick

                # Herd defense bonus: defense aura from tanks
                herd_defense = 0
                nearby_herbs = self.cell_grid.query(prey.x, prey.y, 50)
                for ally in nearby_herbs:
                    if ally.id != prey.id and ally.alive and not ally.genome.is_predator():
                        # Tanks (large + defense) give defense aura
                        if ally.current_size > 12 and ally.genome.defense > 3:
                            herd_defense += ally.genome.defense * 0.2
                herd_defense = min(herd_defense, 4.0)

                # Prey defense reduces damage
                defense = prey.genome.defense + herd_defense
                actual_damage = max(0.05, damage - defense * 0.15)

                # Apply damage
                prey.hp -= actual_damage
                prey.damaged_flash = 8
                prey.being_attacked_by.append(predator.id)
                predator.energy -= predator.current_size * 0.005  # More realistic combat fatigue

                # Bite shock: strong attack -> stun, weaker -> slow
                bite_force = actual_damage / max(prey.max_hp, 1)  # 0-1 ratio
                if bite_force > 0.15:
                    # Strong bite -> short stun (unconscious)
                    prey.stun_ticks = max(prey.stun_ticks, int(5 + bite_force * 20))  # 5-25 ticks
                elif bite_force > 0.05:
                    # Medium bite -> slowdown
                    prey.slow_ticks = max(prey.slow_ticks, int(15 + bite_force * 40))  # 15-55 ticks
                    prey.slow_factor = min(prey.slow_factor, 0.3 + (1.0 - bite_force) * 0.4)  # 30-70% speed

                # --- Knockback: large prey pushes predator back ---
                if prey.current_size > predator.current_size * 1.3:
                    knockback = (prey.current_size - predator.current_size) * 0.15
                    if dist > 0.1:
                        predator.vx -= (dx / dist) * knockback
                        predator.vy -= (dy / dist) * knockback

                # --- Death: prey HP depleted -> CARCASS forms ---
                if prey.hp <= 0:
                    prey.alive = False
                    predator.remember_hunt(prey, True, prey.x, prey.y)
                    predator.kills += 1
                    predator.chase_energy_spent = 0.0
                    predator.chase_target_id = None
                    predator.rest_after_hunt = 0  # Successful hunt -> DON'T rest, eat!
                    self.stats["total_died"] += 1

                    # Instant energy bonus (blood/fresh meat — like in nature)
                    instant_bonus = prey.genome.size * 2.0 + 5.0
                    predator.energy += instant_bonus
                    predator.ticks_without_food = 0
                    predator.last_eat_tick = self.tick

                    # Carcass: size-proportional energy — 5th element: birth tick
                    corpse_energy = prey.energy * 0.5 + prey.genome.size ** 2 * 0.8
                    corpse_energy = max(10, corpse_energy)
                    self.food.append((prey.x, prey.y, corpse_energy, True, self.tick))
                break  # Damages 1 prey at a time


# --- Renderer ---
class Renderer:
    def __init__(self, screen, world_w, world_h):
        self.screen = screen
        self.screen_w = screen.get_width()
        self.screen_h = screen.get_height()
        self.world_w = world_w
        self.world_h = world_h

        self.font_large = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 15)
        self.font_small = pygame.font.SysFont("Arial", 12)

        # Camera
        self.cam_x = 0
        self.cam_y = 0
        self.zoom = 1.0

        self.show_info = True
        self.selected_cell = None
        self.top_genome_cells = []  # Clickable top genome cells [(rect, cell), ...]
        self.show_pop_graph = True  # Population graph visibility
        self.flash_message = ""     # Flash message at screen center
        self.flash_timer = 0        # How many frames it's visible

        # --- Action Camera ---
        self.action_cam = False     # Action camera mode on/off
        self.action_target_x = 0.0  # Current focus point
        self.action_target_y = 0.0
        self.action_timer = 0       # Ticks remaining on current event
        self.action_zoom = 1.8      # Zoom level for action cam
        self.action_label = ""      # What's happening (for HUD)
        self._action_saved_zoom = 1.0  # Saved zoom before action cam

    def world_to_screen(self, wx, wy):
        sx = (wx - self.cam_x) * self.zoom
        sy = (wy - self.cam_y) * self.zoom
        return int(sx), int(sy)

    def screen_to_world(self, sx, sy):
        wx = sx / self.zoom + self.cam_x
        wy = sy / self.zoom + self.cam_y
        return wx, wy

    def draw(self, world, paused, sim_speed):
        self.screen.fill((8, 12, 18))

        # Draw shelters
        for s in world.shelters:
            sx, sy = self.world_to_screen(s['x'], s['y'])
            r = max(5, int(s['radius'] * self.zoom))
            if -r < sx < self.screen_w + r and -r < sy < self.screen_h + r:
                shelter_surf = pygame.Surface((r * 2 + 8, r * 2 + 8), pygame.SRCALPHA)
                cr = r + 4
                # Inner dark area
                pygame.draw.circle(shelter_surf, (40, 35, 25, 70), (cr, cr), r)
                # Outer dashed outline
                pygame.draw.circle(shelter_surf, (100, 80, 50, 90), (cr, cr), r, max(1, int(2 * self.zoom)))
                # Inner pattern — small lines like branches/leaves
                for i in range(6):
                    a = i * math.pi / 3
                    lx = cr + int(math.cos(a) * r * 0.5)
                    ly = cr + int(math.sin(a) * r * 0.5)
                    lx2 = cr + int(math.cos(a) * r * 0.8)
                    ly2 = cr + int(math.sin(a) * r * 0.8)
                    pygame.draw.line(shelter_surf, (80, 65, 40, 50), (lx, ly), (lx2, ly2), max(1, int(self.zoom)))
                self.screen.blit(shelter_surf, (sx - cr, sy - cr))
                # "S" label if large enough
                if r > 12:
                    s_txt = self.font_small.render("S", True, (120, 100, 60))
                    self.screen.blit(s_txt, (sx - s_txt.get_width() // 2, sy - s_txt.get_height() // 2))

        # Draw obstacles (rocks)
        for obs in world.obstacles:
            ox, oy = self.world_to_screen(obs['x'], obs['y'])
            ow = int(obs['w'] * self.zoom)
            oh = int(obs['h'] * self.zoom)
            if ox + ow < 0 or ox > self.screen_w or oy + oh < 0 or oy > self.screen_h:
                continue
            shade = obs['shade']
            base_r, base_g, base_b = int(55 * shade), int(50 * shade), int(45 * shade)
            # Rock body
            obs_surf = pygame.Surface((ow + 4, oh + 4), pygame.SRCALPHA)
            pygame.draw.rect(obs_surf, (base_r, base_g, base_b, 220), (2, 2, ow, oh), border_radius=max(3, ow // 6))
            # Rough border
            pygame.draw.rect(obs_surf, (base_r + 30, base_g + 25, base_b + 20, 180),
                           (2, 2, ow, oh), max(1, int(2 * self.zoom)), border_radius=max(3, ow // 6))
            # Uneven surface — small cracks
            jagged = obs.get('jagged', [1.0] * 8)
            cx, cy = ow // 2 + 2, oh // 2 + 2
            for i in range(len(jagged)):
                a = i * math.pi * 2 / len(jagged)
                rx = int(cx + math.cos(a) * ow * 0.3 * jagged[i])
                ry = int(cy + math.sin(a) * oh * 0.3 * jagged[i])
                rx2 = int(cx + math.cos(a) * ow * 0.15)
                ry2 = int(cy + math.sin(a) * oh * 0.15)
                pygame.draw.line(obs_surf, (base_r + 15, base_g + 12, base_b + 10, 100),
                               (rx2, ry2), (rx, ry), max(1, int(self.zoom)))
            self.screen.blit(obs_surf, (ox - 2, oy - 2))

        # Pheromone heatmap (faint) — ONLY VISIBLE AREA!
        pm = world.pheromones
        gs = pm.grid_size
        # Visible area grid indices (don't iterate over entire world!)
        col_min = max(0, int(self.cam_x / gs) - 1)
        col_max = min(pm.cols - 1, int((self.cam_x + self.screen_w / self.zoom) / gs) + 1)
        row_min = max(0, int(self.cam_y / gs) - 1)
        row_max = min(pm.rows - 1, int((self.cam_y + self.screen_h / self.zoom) / gs) + 1)
        r = max(2, int(gs * 0.4 * self.zoom))
        # Pre-create a surface to reuse
        p_surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        for row_i in range(row_min, row_max + 1):
            for col_i in range(col_min, col_max + 1):
                cell_data = pm.grid[row_i][col_i]
                if not cell_data:
                    continue
                wx = col_i * gs + gs // 2
                wy = row_i * gs + gs // 2
                sx, sy = self.world_to_screen(wx, wy)
                # Collect values
                danger = cell_data.get(PheromoneMap.DANGER, 0)
                trail = cell_data.get(PheromoneMap.TRAIL, 0)
                food_p = cell_data.get(PheromoneMap.FOOD_HERE, 0)
                mate_p = cell_data.get(PheromoneMap.MATE, 0)
                prey_p = cell_data.get(PheromoneMap.PREY_SCENT, 0)
                corpse_p = cell_data.get(PheromoneMap.CORPSE_SCENT, 0)
                total = danger + trail + food_p + mate_p + prey_p + corpse_p
                if total > 0.3:
                    pr = min(80, int(danger * 25 + mate_p * 20 + prey_p * 12 + corpse_p * 10))
                    pg = min(50, int(food_p * 20 + mate_p * 8 + corpse_p * 6))
                    pb = min(50, int(trail * 15 + mate_p * 15))
                    alpha = min(45, int(total * 10))
                    if alpha > 3:
                        p_surf.fill((0, 0, 0, 0))
                        pygame.draw.circle(p_surf, (pr, pg, pb, alpha), (r, r), r)
                        self.screen.blit(p_surf, (sx - r, sy - r))

        # Oases (pastures) — green spot with outline
        for src in world.food_sources:
            sx, sy = self.world_to_screen(src['x'], src['y'])
            r = max(5, int(src['spread'] * 1.5 * self.zoom))
            if -r < sx < self.screen_w + r and -r < sy < self.screen_h + r:
                oasis_surf = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
                cr = r + 2
                # Inner glow — based on richness
                inner_alpha = int(25 + src['richness'] * 25)
                pygame.draw.circle(oasis_surf, (15, 70, 20, inner_alpha), (cr, cr), r)
                # Faint inner core
                inner_r = max(3, r // 2)
                pygame.draw.circle(oasis_surf, (25, 90, 30, inner_alpha + 10), (cr, cr), inner_r)
                # Outline — dotted effect
                outline_alpha = int(40 + src['richness'] * 30)
                pygame.draw.circle(oasis_surf, (40, 110, 35, outline_alpha), (cr, cr), r, max(1, int(self.zoom)))
                self.screen.blit(oasis_surf, (sx - cr, sy - cr))

        # Draw food — size reflects energy
        for food_item in world.food:
            fx, fy, fe = food_item[0], food_item[1], food_item[2]
            is_meat = food_item[3] if len(food_item) > 3 else False
            sx, sy = self.world_to_screen(fx, fy)
            if -5 < sx < self.screen_w + 5 and -5 < sy < self.screen_h + 5:
                if is_meat:
                    # Carcass: color based on freshness — red->brown->greenish (rotten)
                    corpse_tick = food_item[4] if len(food_item) > 4 else 0
                    corpse_age = world.tick - corpse_tick if corpse_tick > 0 else 9999
                    r = max(3, int(math.sqrt(fe) * 0.8 * self.zoom))
                    # Fresh: vivid red, old: dark brown-green
                    decay = min(1.0, corpse_age / 2500.0)
                    red = int(200 * (1 - decay) + 80 * decay)
                    green = int(30 * (1 - decay) + 70 * decay)
                    blue = int(25 * (1 - decay) + 20 * decay)
                    pygame.draw.circle(self.screen, (red, green, blue), (sx, sy), r)
                    # Corpse scent aura outline (cheap, no Surface allocation)
                    if r > 3 and corpse_age > 300:
                        aura_r = r + min(8, int(corpse_age * 0.002 * self.zoom))
                        aura_alpha = int(40 + decay * 50)
                        pygame.draw.circle(self.screen, (min(255, 120 + int(decay * 40)),
                                           90, 40), (sx, sy), aura_r, 1)
                    # Bone outline
                    if r > 4:
                        bone_c = int(150 + 80 * (1 - decay))
                        pygame.draw.circle(self.screen, (bone_c, bone_c // 2, bone_c // 3),
                                         (sx, sy), r, max(1, int(self.zoom)))
                else:
                    # Plant: small grass -> large bush, size-proportional
                    r = max(1, int(math.sqrt(fe) * 0.55 * self.zoom))
                    green = min(255, int(80 + fe * 4))
                    pygame.draw.circle(self.screen, (30 + min(30, int(fe)), green, 30), (sx, sy), r)

        # Pack/herd lines (between nearby allies)
        drawn_links = set()
        for cell in world.cells:
            if not cell.alive or cell.genome.social < 0.4:
                continue
            sx1, sy1 = self.world_to_screen(cell.x, cell.y)
            if not (-50 < sx1 < self.screen_w + 50 and -50 < sy1 < self.screen_h + 50):
                continue
            nearby = world.cell_grid.query(cell.x, cell.y, cell.genome.sense_range * 0.4)
            for other in nearby:
                if other.id == cell.id or not other.alive:
                    continue
                if abs(other.genome.diet - cell.genome.diet) > 0.3:
                    continue
                link_key = (min(cell.id, other.id), max(cell.id, other.id))
                if link_key in drawn_links:
                    continue
                drawn_links.add(link_key)
                sx2, sy2 = self.world_to_screen(other.x, other.y)
                # Shared target = stronger line (pack hunting)
                if cell.target_id and cell.target_id == other.target_id:
                    color = (200, 60, 60, 60)
                else:
                    color = (60, 120, 60, 35)
                link_surf = pygame.Surface((abs(sx2 - sx1) + 2, abs(sy2 - sy1) + 2), pygame.SRCALPHA)
                ox = min(sx1, sx2)
                oy = min(sy1, sy2)
                pygame.draw.line(link_surf, color, (sx1 - ox, sy1 - oy), (sx2 - ox, sy2 - oy), 1)
                self.screen.blit(link_surf, (ox, oy))

        # Draw cells
        for cell in world.cells:
            if not cell.alive:
                continue
            sx, sy = self.world_to_screen(cell.x, cell.y)
            r = max(2, int(cell.radius * self.zoom))

            if -r - 20 < sx < self.screen_w + r + 20 and -r - 20 < sy < self.screen_h + r + 20:
                color = cell.genome.color()

                # Migration indicator (yellow arrow) — for both types
                if cell.migrating:
                    dx = cell.migrate_x - cell.x
                    dy = cell.migrate_y - cell.y
                    d = max(math.sqrt(dx*dx + dy*dy), 1)
                    arrow_len = min(20, d * 0.1) * self.zoom
                    ax = sx + dx / d * arrow_len
                    ay = sy + dy / d * arrow_len
                    col = (220, 200, 50) if not cell.genome.is_predator() else (255, 120, 50)
                    pygame.draw.line(self.screen, col, (sx, sy), (int(ax), int(ay)), max(1, int(self.zoom)))

                # Alpha indicator (white dot on top)
                if cell.leader_id is None and cell.pack_mates >= 2:
                    pygame.draw.circle(self.screen, (255, 255, 255), (sx, sy - r - 3), max(1, int(2 * self.zoom)))

                # Ambush indicator (eye - yellow)
                if cell.ambushing:
                    pygame.draw.circle(self.screen, (200, 180, 50), (sx, sy), r + 2, 1)
                    # Small eye
                    pygame.draw.circle(self.screen, (255, 220, 50), (sx + r // 2, sy - r // 3), max(1, int(self.zoom)))

                # Hiding indicator (fainter drawing)
                if cell.hiding and not cell.genome.is_predator():
                    pygame.draw.circle(self.screen, (60, 80, 60), (sx, sy), r + 1, 1)

                # Sense range + pheromone display (for selected cell)
                if self.selected_cell and self.selected_cell.id == cell.id:
                    sense_r = int(cell.genome.sense_range * self.zoom)
                    if sense_r > 2:
                        sense_surf = pygame.Surface((sense_r * 2, sense_r * 2), pygame.SRCALPHA)
                        pygame.draw.circle(sense_surf, (*color, 20), (sense_r, sense_r), sense_r)
                        pygame.draw.circle(sense_surf, (*color, 35), (sense_r, sense_r), sense_r, 1)
                        self.screen.blit(sense_surf, (sx - sense_r, sy - sense_r))

                    # Pheromone heatmap highlight around cell
                    self._draw_selected_pheromones(world, cell)

                # --- Draw cilia ---
                cilia_angles = cell.genome.cilia_positions()
                cilia_len = max(3, int((r * 0.6 + 2) * self.zoom))
                for i, ca in enumerate(cilia_angles):
                    abs_angle = cell.angle + ca
                    # Cilia wave animation
                    wave = math.sin(cell.cilia_phase + i * 1.5) * 0.3
                    draw_angle = abs_angle + wave

                    # Cilia starting point on cell surface
                    cx1 = sx + int(math.cos(abs_angle) * r)
                    cy1 = sy + int(math.sin(abs_angle) * r)
                    # Cilia endpoint
                    cx2 = cx1 + int(math.cos(draw_angle) * cilia_len)
                    cy2 = cy1 + int(math.sin(draw_angle) * cilia_len)

                    cilia_color = tuple(min(255, c + 80) for c in color)
                    pygame.draw.line(self.screen, cilia_color, (cx1, cy1), (cx2, cy2),
                                     max(1, int(self.zoom)))

                # Cell body — stealthy predators are fainter
                if cell.genome.is_predator() and cell.thrust < 0.35 and not cell.sprinting:
                    spd = math.sqrt(cell.vx * cell.vx + cell.vy * cell.vy)
                    stealth_alpha = max(60, int(255 * (0.3 + 0.7 * min(spd / max(cell.genome.max_speed * 0.5, 0.1), 1.0))))
                    stealth_surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                    pygame.draw.circle(stealth_surf, (*color, stealth_alpha), (r, r), r)
                    self.screen.blit(stealth_surf, (sx - r, sy - r))
                else:
                    pygame.draw.circle(self.screen, color, (sx, sy), r)

                # Inner core (nucleus)
                inner_r = max(1, r // 2)
                brighter = tuple(min(255, c + 50) for c in color)
                pygame.draw.circle(self.screen, brighter, (sx, sy), inner_r)

                # Direction indicator dot (where it's looking)
                if r > 4:
                    dir_x = sx + int(math.cos(cell.angle) * r * 0.6)
                    dir_y = sy + int(math.sin(cell.angle) * r * 0.6)
                    pygame.draw.circle(self.screen, (255, 255, 255), (dir_x, dir_y), max(1, r // 5))

                # Predator: red border, Omnivore: orange border
                if cell.genome.is_predator():
                    pygame.draw.circle(self.screen, (255, 60, 60), (sx, sy), r, 1)
                elif cell.genome.is_omnivore():
                    pygame.draw.circle(self.screen, (255, 180, 40), (sx, sy), r, 1)

                # Hibernation visual: gray, pulsating circle
                if cell.hibernating:
                    pulse = abs(math.sin(cell.age * 0.02)) * 0.5 + 0.5
                    hib_r = int(r + 2 + pulse * 2)
                    hib_surf = pygame.Surface((hib_r * 2, hib_r * 2), pygame.SRCALPHA)
                    pygame.draw.circle(hib_surf, (100, 100, 120, int(60 + pulse * 40)),
                                     (hib_r, hib_r), hib_r, max(1, int(2 * self.zoom)))
                    self.screen.blit(hib_surf, (sx - hib_r, sy - hib_r))
                    # "Zzz" indicator
                    if r > 5:
                        zzz = self.font_small.render("z", True, (150, 150, 180))
                        self.screen.blit(zzz, (sx + r, sy - r - 5))

                # Sprint effect: flashing bright ring
                if cell.sprinting:
                    sprint_color = (255, 255, 100) if cell.genome.is_predator() else (100, 255, 255)
                    pygame.draw.circle(self.screen, sprint_color, (sx, sy), r + 3, 1)

                # Damage flash
                if cell.damaged_flash > 0:
                    flash_alpha = min(180, cell.damaged_flash * 25)
                    flash_surf = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
                    pygame.draw.circle(flash_surf, (255, 50, 30, flash_alpha), (r + 2, r + 2), r + 2)
                    self.screen.blit(flash_surf, (sx - r - 2, sy - r - 2))

                # Juvenile indicator (smaller, fainter contour)
                if cell.is_child:
                    pygame.draw.circle(self.screen, (180, 180, 255), (sx, sy), r + 2, 1)

                # Mate-seeking indicator (pink heart-like pulsating circle)
                if cell.seeking_mate and r > 3:
                    pulse = abs(math.sin(cell.age * 0.08)) * 0.5 + 0.5
                    mate_r = int(r + 3 + pulse * 3)
                    mate_surf = pygame.Surface((mate_r * 2, mate_r * 2), pygame.SRCALPHA)
                    pygame.draw.circle(mate_surf, (255, 100, 180, int(40 + pulse * 40)),
                                     (mate_r, mate_r), mate_r, max(1, int(2 * self.zoom)))
                    self.screen.blit(mate_surf, (sx - mate_r, sy - mate_r))

                # === TAUNT visual effects (sonar ping) ===
                if cell.taunt_timer > 0 and cell.taunt_type:
                    tp = cell.genome.taunt_power
                    max_timer = int(30 + tp * 15)
                    if cell.taunt_type == 'mate' and cell.genome.is_predator():
                        max_timer *= 2
                    t_progress = 1.0 - cell.taunt_timer / max(max_timer, 1)  # 0->1
                    wave_r = int(cell.taunt_radius * self.zoom)
                    # Stronger gene = brighter wave
                    base_alpha = int(100 + tp * 60)  # 100-250
                    t_alpha = max(0, min(255, int(base_alpha * (1.0 - t_progress))))
                    # Line width depends on gene strength
                    line_w = max(1, int((1 + tp) * self.zoom))

                    if cell.taunt_type == 'mate':
                        # Pink sonar ping
                        if wave_r > 2:
                            t_surf = pygame.Surface((wave_r * 2, wave_r * 2), pygame.SRCALPHA)
                            pygame.draw.circle(t_surf, (255, 80, 200, t_alpha),
                                             (wave_r, wave_r), wave_r, line_w)
                            self.screen.blit(t_surf, (sx - wave_r, sy - wave_r))
                        if r > 3:
                            heart = self.font_small.render("♥", True,
                                (255, 100, 180, min(255, t_alpha + 80)))
                            self.screen.blit(heart, (sx - 4, sy - r - 14))

                    elif cell.taunt_type == 'attack':
                        # Red sonar ping
                        if wave_r > 2:
                            t_surf = pygame.Surface((wave_r * 2, wave_r * 2), pygame.SRCALPHA)
                            pygame.draw.circle(t_surf, (255, 40, 40, t_alpha),
                                             (wave_r, wave_r), wave_r, line_w)
                            self.screen.blit(t_surf, (sx - wave_r, sy - wave_r))
                        if r > 3:
                            excl = self.font_medium.render("!", True, (255, 60, 60))
                            self.screen.blit(excl, (sx - 3, sy - r - 18))

                    elif cell.taunt_type == 'flee':
                        # Yellow sonar ping
                        if wave_r > 2:
                            t_surf = pygame.Surface((wave_r * 2, wave_r * 2), pygame.SRCALPHA)
                            pygame.draw.circle(t_surf, (255, 200, 40, t_alpha),
                                             (wave_r, wave_r), wave_r, line_w)
                            self.screen.blit(t_surf, (sx - wave_r, sy - wave_r))
                        if r > 3:
                            warn = self.font_medium.render("!!", True, (255, 220, 50))
                            self.screen.blit(warn, (sx - 6, sy - r - 18))

                # HP bar (top, red)
                if r > 4:
                    bar_w = r * 2
                    bar_h = 2
                    hp_ratio = cell.hp_ratio
                    if hp_ratio < 1.0:  # Only if damaged
                        pygame.draw.rect(self.screen, (40, 40, 40),
                                         (sx - bar_w // 2, sy - r - 9, bar_w, bar_h))
                        hp_color = (200, 50, 50) if hp_ratio < 0.3 else (220, 160, 30) if hp_ratio < 0.6 else (50, 200, 50)
                        pygame.draw.rect(self.screen, hp_color,
                                         (sx - bar_w // 2, sy - r - 9, int(bar_w * hp_ratio), bar_h))

                    # Energy bar (below, green)
                    energy_ratio = min(1, cell.energy / max(1, cell.genome.repro_thresh))
                    pygame.draw.rect(self.screen, (30, 30, 30),
                                     (sx - bar_w // 2, sy - r - 6, bar_w, bar_h))
                    e_color = (50, 180, 50) if energy_ratio > 0.3 else (180, 50, 50)
                    pygame.draw.rect(self.screen, e_color,
                                     (sx - bar_w // 2, sy - r - 6, int(bar_w * energy_ratio), bar_h))

        # Selected cell info
        if self.selected_cell and self.selected_cell.alive:
            self._draw_cell_info(self.selected_cell)

        # HUD
        self._draw_hud(world, paused, sim_speed)

    def _draw_hud(self, world, paused, sim_speed):
        # Top bar
        bar_h = 30
        bar_surf = pygame.Surface((self.screen_w, bar_h), pygame.SRCALPHA)
        bar_surf.fill((0, 0, 0, 160))
        self.screen.blit(bar_surf, (0, 0))

        # Info
        alive = len(world.cells)
        pred = world.stats["predators"]
        herb = world.stats["herbivores"]
        omni = world.stats["omnivores"]
        food = len(world.food)
        gen = world.stats["max_generation"]

        status = "PAUSED" if paused else f"x{sim_speed}"
        info = (f"Tick: {world.tick:,}  |  Cells: {alive} "
                f"(H:{herb} O:{omni} P:{pred})  |  "
                f"Food: {food}  |  Gen: {gen}  |  {status}")

        text = self.font_medium.render(info, True, (200, 220, 200))
        self.screen.blit(text, (10, 6))

        # Bottom bar - controls
        bot_y = self.screen_h - 24
        bot_surf = pygame.Surface((self.screen_w, 24), pygame.SRCALPHA)
        bot_surf.fill((0, 0, 0, 140))
        self.screen.blit(bot_surf, (0, bot_y))

        controls = "SPACE: Pause | W/S: Speed | C: Action Cam | H: Headless | Ctrl+S/L: Save/Load | 1/2/3: Spawn | F: Food | M: Settings | R: Reset"
        ct = self.font_small.render(controls, True, (140, 140, 160))
        self.screen.blit(ct, (10, bot_y + 5))

        # Right side info panel
        if self.show_info:
            self._draw_stats_panel(world)

        # Top genomes panel (bottom right corner)
        self._draw_top_genomes(world)

        # Minimap (bottom left, only in action cam mode)
        if self.action_cam:
            self._draw_minimap(world)

        # Population graph (bottom center)
        if self.show_pop_graph and len(world.pop_history) > 2:
            self._draw_pop_graph(world)

        # Flash message (save/load feedback)
        if self.flash_timer > 0:
            self.flash_timer -= 1
            alpha = min(255, self.flash_timer * 4)
            msg_surf = self.font_large.render(self.flash_message, True, (255, 255, 100))
            msg_w = msg_surf.get_width()
            msg_h = msg_surf.get_height()
            bg = pygame.Surface((msg_w + 20, msg_h + 12), pygame.SRCALPHA)
            bg.fill((0, 0, 0, min(180, alpha)))
            bx = (self.screen_w - msg_w - 20) // 2
            by = self.screen_h // 2 - 40
            self.screen.blit(bg, (bx, by))
            msg_surf.set_alpha(alpha)
            self.screen.blit(msg_surf, (bx + 10, by + 6))

    def _draw_stats_panel(self, world):
        pw, ph = 220, 200
        px = self.screen_w - pw - 10
        py = 40

        panel_surf = pygame.Surface((pw, ph), pygame.SRCALPHA)
        panel_surf.fill((0, 0, 0, 180))
        self.screen.blit(panel_surf, (px, py))
        pygame.draw.rect(self.screen, (60, 80, 100), (px, py, pw, ph), 1, border_radius=4)

        y = py + 8
        title = self.font_medium.render("Statistics", True, (100, 200, 255))
        self.screen.blit(title, (px + 10, y))
        y += 22

        stats_lines = [
            f"Born: {world.stats['total_born']:,}",
            f"Died: {world.stats['total_died']:,}",
            f"Food consumed: {world.stats['total_eaten']:,}",
            f"Max generation: {world.stats['max_generation']}",
            "",
            "Average genome:",
        ]

        for line in stats_lines:
            t = self.font_small.render(line, True, (180, 180, 190))
            self.screen.blit(t, (px + 10, y))
            y += 16

        # Average genome values
        if world.cells:
            avg_size = sum(c.genome.size for c in world.cells) / len(world.cells)
            avg_spd = sum(c.genome.max_speed for c in world.cells) / len(world.cells)
            avg_sense = sum(c.genome.sense_range for c in world.cells) / len(world.cells)
            avg_cilia = sum(c.genome.num_cilia for c in world.cells) / len(world.cells)
            avg_spread = sum(c.genome.cilia_spread for c in world.cells) / len(world.cells)

            genome_lines = [
                f"  Size: {avg_size:.1f}  MaxSpd: {avg_spd:.1f}",
                f"  Sense: {avg_sense:.0f}  Cilia: {avg_cilia:.1f}",
                f"  Spread: {avg_spread:.0%}",
            ]
            for line in genome_lines:
                t = self.font_small.render(line, True, (160, 180, 160))
                self.screen.blit(t, (px + 10, y))
                y += 16

    def _draw_top_genomes(self, world):
        """Top genomes panel — bottom right corner, clickable. Cached: refreshes every 30 frames."""
        if not world.cells:
            return

        pw = 240
        line_h = 16
        header_h = 20
        cells_per_type = 3
        type_block = header_h + cells_per_type * line_h + 4
        total_h = type_block * 3 + 16
        px = self.screen_w - pw - 10
        py = self.screen_h - total_h - 32

        # Cache: only recalculate top list every 30 ticks
        cache_key = world.tick // 30
        if not hasattr(self, '_top_cache_key') or self._top_cache_key != cache_key:
            self._top_cache_key = cache_key
            self._top_cache = {}
            for type_name, diet_check in [("herb", "is_herbivore"), ("omni", "is_omnivore"), ("pred", "is_predator")]:
                type_cells = [c for c in world.cells if c.alive and getattr(c.genome, diet_check)()]
                type_cells.sort(key=lambda c: c.calc_fitness(), reverse=True)
                self._top_cache[type_name] = type_cells[:cells_per_type]

        # Panel background
        panel_surf = pygame.Surface((pw, total_h), pygame.SRCALPHA)
        panel_surf.fill((0, 0, 0, 180))
        self.screen.blit(panel_surf, (px, py))
        pygame.draw.rect(self.screen, (60, 80, 100), (px, py, pw, total_h), 1, border_radius=4)

        title = self.font_small.render("Top Genomes", True, (100, 200, 255))
        self.screen.blit(title, (px + pw // 2 - title.get_width() // 2, py + 2))

        self.top_genome_cells = []
        y = py + 18
        mx, my = pygame.mouse.get_pos()

        types = [
            ("Herbiv.", "herb", (80, 200, 80)),
            ("Omniv.", "omni", (220, 160, 50)),
            ("Predator", "pred", (220, 70, 70)),
        ]

        for type_name, cache_key_name, color in types:
            header = self.font_small.render(type_name, True, color)
            self.screen.blit(header, (px + 8, y))
            y += header_h - 2

            top = [c for c in self._top_cache.get(cache_key_name, []) if c.alive]

            for cell in top:
                is_selected = (self.selected_cell and self.selected_cell.id == cell.id)
                txt_color = (255, 255, 100) if is_selected else (190, 195, 210)
                prefix = ">" if is_selected else " "
                info = f"{prefix}#{cell.id} E:{cell.energy:.0f} G:{cell.generation} C:{cell.children}"
                t = self.font_small.render(info, True, txt_color)
                rect = pygame.Rect(px + 4, y, pw - 8, line_h)
                self.screen.blit(t, (px + 8, y))
                if rect.collidepoint(mx, my):
                    hover_surf = pygame.Surface((pw - 8, line_h), pygame.SRCALPHA)
                    hover_surf.fill((80, 120, 180, 50))
                    self.screen.blit(hover_surf, (px + 4, y))
                self.top_genome_cells.append((rect, cell))
                y += line_h

            if not top:
                t = self.font_small.render("  -", True, (100, 100, 110))
                self.screen.blit(t, (px + 8, y))
                y += line_h * cells_per_type

            y += 4

    def _draw_pop_graph(self, world):
        """Population graph — cached surface, only redrawn when data changes."""
        history = world.pop_history
        n = len(history)
        if n < 2:
            return

        gw, gh = 300, 100
        gx = (self.screen_w - gw) // 2
        gy = self.screen_h - gh - 32

        # Cache: only redraw on new data
        if not hasattr(self, '_pop_graph_cache') or self._pop_graph_n != n:
            self._pop_graph_n = n
            self._pop_graph_surf = pygame.Surface((gw, gh), pygame.SRCALPHA)
            surf = self._pop_graph_surf
            surf.fill((0, 0, 0, 170))
            pygame.draw.rect(surf, (50, 65, 85), (0, 0, gw, gh), 1, border_radius=3)

            title = self.font_small.render("Population", True, (100, 200, 255))
            surf.blit(title, (gw // 2 - title.get_width() // 2, 2))

            max_pop = max(max(h, o, p) for _, h, o, p in history)
            max_pop = max(1, max_pop)
            draw_x, draw_y = 5, 16
            draw_w, draw_h = gw - 10, gh - 22

            colors = [(80, 200, 80), (220, 160, 50), (220, 70, 70)]
            for species_idx, color in zip([1, 2, 3], colors):
                points = []
                for i, entry in enumerate(history):
                    x = draw_x + int(i * draw_w / (n - 1))
                    val = entry[species_idx]
                    y_pos = draw_y + draw_h - int(val * draw_h / max_pop)
                    points.append((x, y_pos))
                if len(points) >= 2:
                    pygame.draw.lines(surf, color, False, points, 2)

            legend_y = gh - 14
            legends = [("H", (80, 200, 80)), ("O", (220, 160, 50)), ("P", (220, 70, 70))]
            lx = 10
            for label, color in legends:
                pygame.draw.line(surf, color, (lx, legend_y + 4), (lx + 12, legend_y + 4), 2)
                lt = self.font_small.render(label, True, color)
                surf.blit(lt, (lx + 15, legend_y - 2))
                lx += 40

            max_t = self.font_small.render(str(max_pop), True, (140, 140, 160))
            surf.blit(max_t, (gw - max_t.get_width() - 4, draw_y - 2))
            self._pop_graph_cache = True

        self.screen.blit(self._pop_graph_surf, (gx, gy))

    def _draw_selected_pheromones(self, world, cell):
        """Strong display of pheromones around selected cell."""
        pm = world.pheromones
        gs = pm.grid_size
        sense = cell.genome.sense_range
        # Look at pheromones in sense radius circle from cell position
        col_min = max(0, int((cell.x - sense) / gs))
        col_max = min(pm.cols - 1, int((cell.x + sense) / gs))
        row_min = max(0, int((cell.y - sense) / gs))
        row_max = min(pm.rows - 1, int((cell.y + sense) / gs))

        r = max(3, int(gs * 0.45 * self.zoom))
        p_surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        for row_i in range(row_min, row_max + 1):
            for col_i in range(col_min, col_max + 1):
                cell_data = pm.grid[row_i][col_i]
                if not cell_data:
                    continue
                wx = col_i * gs + gs // 2
                wy = row_i * gs + gs // 2
                sx, sy = self.world_to_screen(wx, wy)
                if -r < sx < self.screen_w + r and -r < sy < self.screen_h + r:
                    # Select strongest pheromone (1 surface/grid -> 7x fewer blits)
                    best_val = 0
                    best_col = (100, 100, 100)
                    for ptype, col in [
                        (PheromoneMap.DANGER, (255, 50, 30)),
                        (PheromoneMap.TRAIL, (60, 120, 255)),
                        (PheromoneMap.FOOD_HERE, (50, 220, 50)),
                        (PheromoneMap.MATE, (255, 100, 200)),
                        (PheromoneMap.PREY_SCENT, (255, 160, 40)),
                        (PheromoneMap.AMBUSH, (200, 200, 50)),
                        (PheromoneMap.CORPSE_SCENT, (160, 100, 40)),
                    ]:
                        v = cell_data.get(ptype, 0)
                        if v > best_val:
                            best_val = v
                            best_col = col
                    if best_val > 0.1:
                        alpha = min(120, int(best_val * 60))
                        p_surf.fill((0, 0, 0, 0))
                        pygame.draw.circle(p_surf, (*best_col, alpha), (r, r), r)
                        self.screen.blit(p_surf, (sx - r, sy - r))

    def _draw_cell_info(self, cell):
        pw, ph = 270, 450
        px = 10
        py = 40

        panel_surf = pygame.Surface((pw, ph), pygame.SRCALPHA)
        panel_surf.fill((0, 0, 0, 200))
        self.screen.blit(panel_surf, (px, py))

        color = cell.genome.color()
        pygame.draw.rect(self.screen, color, (px, py, pw, ph), 1, border_radius=4)

        y = py + 8
        kind = "Predator" if cell.genome.is_predator() else ("Omnivore" if cell.genome.is_omnivore() else "Herbivore")
        if cell.hibernating:
            kind += " [HIBERNATING]"
        title = self.font_medium.render(f"Cell #{cell.id} ({kind})", True, color)
        self.screen.blit(title, (px + 10, y))
        y += 20

        # Cilia type description
        n = cell.genome.num_cilia
        sp = cell.genome.cilia_spread
        if sp < 0.2:
            cilia_desc = f"{n} cilia, unidirectional (torpedo)"
        elif sp < 0.5:
            cilia_desc = f"{n} cilia, narrow spread"
        elif sp < 0.8:
            cilia_desc = f"{n} cilia, wide spread"
        else:
            cilia_desc = f"{n} cilia, radial (jellyfish)"

        child_str = " [JUVENILE]" if cell.is_child else (" [SEEKING MATE]" if cell.seeking_mate else "")
        lines = [
            f"HP: {cell.hp:.0f}/{cell.max_hp:.0f}  ({cell.hp_ratio:.0%}){child_str}",
            f"Energy: {cell.energy:.0f} / {cell.genome.repro_thresh:.0f}",
            f"Age: {cell.age}  Gen: {cell.generation}",
            f"Children: {cell.children}  Kills: {cell.kills}",
            f"Damage/tick: {cell.damage_per_tick:.2f}",
            "",
            f"--- Cilia system ---",
            f"  {cilia_desc}",
            f"  Power/cilium: {cell.genome.cilia_power:.2f}",
            f"  Speed: {cell.genome.max_speed:.2f} x{cell.effective_speed_mult:.0%}" + (f"  Alert:{cell.alertness:.0%}" if not cell.genome.is_predator() else ""),
            f"  Turn rate: {math.degrees(cell.genome.turn_rate):.1f}\u00b0/tick",
            f"  Maneuver: {cell.genome.maneuverability:.0%}",
            "",
            f"Size: {cell.current_size:.1f} (max:{cell.genome.size:.1f})",
            f"Sense range: {cell.genome.sense_range:.0f}",
            f"Attack: {cell.genome.attack:.1f}  Defense: {cell.genome.defense:.1f}",
            f"Metabolism: {cell.genome.metabolism:.2f}",
            f"Diet: {cell.genome.diet:.2f}",
            f"Aggression: {cell.genome.aggression:.2f}  Social: {cell.genome.social:.2f}",
            f"Taunt: {cell.genome.taunt_power:.2f}  Stealth: {cell.genome.stealth:.2f}",
            f"Litter: {cell.genome.litter_size}  Sprint: {cell.sprint_energy:.0f}% {'ACTIVE' if cell.sprinting else 'ready' if cell.sprint_cooldown <= 0 else f'cooldown:{cell.sprint_cooldown}'}",
            f"Allies: {cell.pack_mates}  Leader: {'ALPHA' if cell.leader_id is None and cell.pack_mates > 0 else f'#{cell.leader_id}' if cell.leader_id else '-'}  ({cell.leadership:.1f}pt)",
            f"Starvation: {cell.ticks_without_food} tick" + (" | MIGRATING" if cell.migrating else "") + (" | AMBUSH" if cell.ambushing else "") + (" | HIDING" if cell.hiding and not cell.ambushing else ""),
        ]
        # Predator memory display
        if cell.genome.is_predator() and (cell.prey_memory or cell.hunting_spots):
            lines.append("")
            lines.append("--- Memory ---")
            if cell.prey_memory:
                s = sum(1 for m in cell.prey_memory if m[3])
                f_count = len(cell.prey_memory) - s
                lines.append(f"  Hunts: {s} successful / {f_count} failed")
            if cell.prefer_weak != 0 or cell.prefer_isolated != 0:
                prefs = []
                if cell.prefer_weak > 0.2: prefs.append("seeks weak")
                if cell.prefer_isolated > 0.2: prefs.append("seeks isolated")
                if cell.prefer_slow > 0.2: prefs.append("seeks slow")
                if prefs:
                    lines.append(f"  Learned: {', '.join(prefs)}")
            if cell.hunting_spots:
                lines.append(f"  Hunting grounds: {len(cell.hunting_spots)}")
        # Herbivore memory
        if not cell.genome.is_predator():
            has_mem = cell.food_spots or cell.danger_zones or cell.migrating
            if has_mem:
                lines.append("")
                lines.append("--- Memory ---")
                if cell.food_spots:
                    lines.append(f"  Known pastures: {len(cell.food_spots)}")
                if cell.danger_zones:
                    lines.append(f"  Danger zones: {len(cell.danger_zones)}")
                if cell.migrating:
                    dx = cell.migrate_x - cell.x
                    dy = cell.migrate_y - cell.y
                    md = math.sqrt(dx*dx + dy*dy)
                    lines.append(f"  MIGRATING -> ({cell.migrate_x:.0f},{cell.migrate_y:.0f}) [{md:.0f}px]")
                if cell.attacks_here > 0:
                    lines.append(f"  Attacks here: {cell.attacks_here}")

        # Pheromone legend
        lines.append("")
        lines.append("--- Pheromones ---")

        for line in lines:
            col = (140, 200, 255) if line.startswith("---") else (190, 190, 200)
            t = self.font_small.render(line, True, col)
            self.screen.blit(t, (px + 10, y))
            y += 15

        # Pheromone color legend
        legend = [
            ((255, 50, 30), "Danger"),
            ((60, 120, 255), "Trail"),
            ((50, 220, 50), "Food"),
            ((255, 100, 200), "Mate"),
            ((255, 160, 40), "Prey scent"),
            ((200, 200, 50), "Ambush"),
            ((160, 100, 40), "Corpse scent"),
        ]
        lx = px + 15
        for col, name in legend:
            pygame.draw.circle(self.screen, col, (lx, y + 6), 4)
            t = self.font_small.render(name, True, (180, 180, 190))
            self.screen.blit(t, (lx + 8, y))
            lx += 8 + t.get_width() + 8
            if lx > px + pw - 20:
                lx = px + 15
                y += 15
        y += 15

    def update_action_cam(self, world):
        """Find the most exciting event and smoothly move camera there."""
        if not self.action_cam:
            return

        self.action_timer -= 1

        # Still watching current event — smooth camera pan
        if self.action_timer > 0:
            target_cam_x = self.action_target_x - self.screen_w / (2 * self.action_zoom)
            target_cam_y = self.action_target_y - self.screen_h / (2 * self.action_zoom)
            # Smooth interpolation (lerp)
            self.cam_x += (target_cam_x - self.cam_x) * 0.06
            self.cam_y += (target_cam_y - self.cam_y) * 0.06
            self.zoom += (self.action_zoom - self.zoom) * 0.06
            return

        # Timer expired — find new event
        best_score = -1
        best_x, best_y = self.action_target_x, self.action_target_y
        best_label = ""
        best_zoom = 1.8

        for cell in world.cells:
            if not cell.alive:
                continue
            score = 0
            label = ""

            # Active combat (highest priority)
            if cell.being_attacked_by:
                score = 10
                label = "HUNT"
                best_zoom = 2.2

            # Predator sprinting at prey
            elif cell.genome.is_predator() and cell.sprinting and cell.target_id:
                score = 7
                label = "CHASE"
                best_zoom = 1.8

            # Pack hunting (multiple predators on same target)
            elif cell.genome.is_predator() and cell.target_id:
                pack = sum(1 for c in world.cells if c.alive and c.genome.is_predator()
                           and c.target_id == cell.target_id)
                if pack >= 2:
                    score = 8
                    label = f"PACK HUNT x{pack}"
                    best_zoom = 1.5

            # Flee taunt (panic)
            elif cell.taunt_type == 'flee':
                score = 5
                label = "PANIC"
                best_zoom = 1.6

            # Attack taunt from alpha
            elif cell.taunt_type == 'attack':
                score = 6
                label = "ALPHA CALL"
                best_zoom = 1.5

            # Mating
            elif cell.seeking_mate and cell.taunt_type == 'mate':
                score = 3
                label = "MATING CALL"
                best_zoom = 2.0

            # Hibernation wake-up
            elif not cell.hibernating and cell.has_hibernated and cell.sprinting and cell.age < 10:
                score = 4
                label = "WAKEUP"
                best_zoom = 2.2

            # Add distance penalty (prefer events far from current view for variety)
            dx = cell.x - self.action_target_x
            dy = cell.y - self.action_target_y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 200:
                score += 1  # Bonus for variety

            if score > best_score:
                best_score = score
                best_x = cell.x
                best_y = cell.y
                best_label = label
                best_zoom_final = best_zoom

        if best_score > 0:
            self.action_target_x = best_x
            self.action_target_y = best_y
            self.action_label = best_label
            self.action_zoom = best_zoom_final
            # Stay 5-10 seconds (300-600 ticks at 60fps)
            self.action_timer = 300 + int(best_score * 30)  # More exciting = longer
        else:
            # Nothing exciting — random roam
            self.action_target_x = random.uniform(200, world.width - 200)
            self.action_target_y = random.uniform(200, world.height - 200)
            self.action_label = "ROAMING"
            self.action_zoom = 1.2
            self.action_timer = 180  # 3 seconds

    def _draw_minimap(self, world):
        """Minimap in bottom-left corner showing entire world."""
        mm_w, mm_h = 180, 180
        mx, my = 10, self.screen_h - mm_h - 32
        scale_x = mm_w / world.width
        scale_y = mm_h / world.height

        # Background
        mm_surf = pygame.Surface((mm_w, mm_h), pygame.SRCALPHA)
        mm_surf.fill((0, 0, 0, 180))

        # Food sources (oases) as green dots
        for src in world.food_sources:
            ox = int(src['x'] * scale_x)
            oy = int(src['y'] * scale_y)
            r = max(2, int(src['spread'] * scale_x * 0.5))
            pygame.draw.circle(mm_surf, (30, 80, 30, 100), (ox, oy), r)

        # Obstacles as dark rectangles
        for obs in world.obstacles:
            ox = int(obs['x'] * scale_x)
            oy = int(obs['y'] * scale_y)
            ow = max(2, int(obs['w'] * scale_x))
            oh = max(2, int(obs['h'] * scale_y))
            pygame.draw.rect(mm_surf, (50, 45, 40, 160), (ox, oy, ow, oh))

        # Cells as colored dots
        for cell in world.cells:
            if not cell.alive:
                continue
            cx = int(cell.x * scale_x)
            cy = int(cell.y * scale_y)
            if cell.genome.is_predator():
                col = (220, 60, 60)
            elif cell.genome.is_omnivore():
                col = (220, 160, 50)
            else:
                col = (60, 200, 60)
            r = max(1, int(cell.current_size * scale_x * 0.5))
            pygame.draw.circle(mm_surf, col, (cx, cy), r)

        # Camera viewport rectangle
        vx = int(self.cam_x * scale_x)
        vy = int(self.cam_y * scale_y)
        vw = max(4, int(self.screen_w / self.zoom * scale_x))
        vh = max(4, int(self.screen_h / self.zoom * scale_y))
        pygame.draw.rect(mm_surf, (255, 255, 255, 120), (vx, vy, vw, vh), 1)

        # Border
        pygame.draw.rect(mm_surf, (80, 100, 130), (0, 0, mm_w, mm_h), 1)

        self.screen.blit(mm_surf, (mx, my))

        # Action label
        if self.action_label:
            label_surf = self.font_medium.render(
                f"ACTION CAM: {self.action_label}", True, (255, 200, 80))
            self.screen.blit(label_surf, (mx, my - 18))

    def handle_click(self, pos, world):
        # First: click on top genomes panel?
        mx, my = pos
        for rect, cell in self.top_genome_cells:
            if rect.collidepoint(mx, my) and cell.alive:
                self.selected_cell = cell
                # Camera focus on cell
                self.cam_x = cell.x - self.screen_w / (2 * self.zoom)
                self.cam_y = cell.y - self.screen_h / (2 * self.zoom)
                return

        # Click on world: cell selection
        wx, wy = self.screen_to_world(*pos)
        best = None
        best_dist = 30

        for cell in world.cells:
            if not cell.alive:
                continue
            dx = cell.x - wx
            dy = cell.y - wy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < best_dist:
                best = cell
                best_dist = dist

        self.selected_cell = best


# --- Settings menu ---
class SettingsMenu:
    """Live settings panel with sliders."""

    ITEMS = [
        {"name": "Food energy",        "key": "food_energy",      "min": 5,     "max": 60,   "step": 1,     "fmt": "{:.0f}"},
        {"name": "Eating speed",       "key": "eat_speed",        "min": 0.2,   "max": 3.0,  "step": 0.1,   "fmt": "{:.1f}"},
        {"name": "Food regrowth",      "key": "food_regrow",      "min": 0.005, "max": 0.15, "step": 0.005, "fmt": "{:.3f}"},
        {"name": "Reproduction cost",  "key": "repro_cost",       "min": 0.15,  "max": 0.8,  "step": 0.05,  "fmt": "{:.0%}"},
        {"name": "Mutation rate",      "key": "mutation_rate",    "min": 0.02,  "max": 0.5,  "step": 0.02,  "fmt": "{:.0%}"},
        {"name": "Mutation strength",  "key": "mutation_strength", "min": 0.02,  "max": 0.4,  "step": 0.02,  "fmt": "{:.0%}"},
        {"name": "Energy drain",      "key": "energy_drain",      "min": 0.2,   "max": 3.0,  "step": 0.1,   "fmt": "{:.1f}x"},
        {"name": "Predator sense",   "key": "pred_sense",        "min": 0.3,   "max": 3.0,  "step": 0.1,   "fmt": "{:.1f}x"},
        {"name": "Max population",    "key": "max_pop",           "min": 50,    "max": 1000, "step": 50,    "fmt": "{:.0f}"},
    ]

    def __init__(self, screen_w, screen_h):
        self.visible = False
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.panel_w = 340
        self.row_h = 36
        self.header_h = 40
        self.panel_h = self.header_h + len(self.ITEMS) * self.row_h + 20
        self.panel_x = (screen_w - self.panel_w) // 2
        self.panel_y = (screen_h - self.panel_h) // 2
        self.font = pygame.font.SysFont("Arial", 14)
        self.font_title = pygame.font.SysFont("Arial", 18, bold=True)
        self.hovered_item = -1

    def toggle(self):
        self.visible = not self.visible

    def draw(self, screen, settings):
        if not self.visible:
            return
        # Background
        overlay = pygame.Surface((self.screen_w, self.screen_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        screen.blit(overlay, (0, 0))

        px, py = self.panel_x, self.panel_y
        pw, ph = self.panel_w, self.panel_h

        # Panel background
        panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
        panel.fill((20, 25, 35, 240))
        screen.blit(panel, (px, py))
        pygame.draw.rect(screen, (80, 120, 180), (px, py, pw, ph), 2, border_radius=6)

        # Title
        title = self.font_title.render("Settings  (M: close)", True, (100, 200, 255))
        screen.blit(title, (px + 15, py + 10))

        # Rows
        for i, item in enumerate(self.ITEMS):
            row_y = py + self.header_h + i * self.row_h
            val = settings.get(item['key'], 0)

            # Hover highlight
            if i == self.hovered_item:
                highlight = pygame.Surface((pw - 4, self.row_h - 2), pygame.SRCALPHA)
                highlight.fill((50, 70, 100, 80))
                screen.blit(highlight, (px + 2, row_y))

            # Name
            name_surf = self.font.render(item['name'], True, (190, 195, 210))
            screen.blit(name_surf, (px + 15, row_y + 4))

            # Value
            val_str = item['fmt'].format(val)
            val_surf = self.font.render(val_str, True, (255, 240, 150))
            screen.blit(val_surf, (px + 195, row_y + 4))

            # [-] button
            btn_y = row_y + 3
            btn_w, btn_h = 28, 22
            minus_x = px + pw - 80
            plus_x = px + pw - 42

            # Slider background bar
            bar_x = px + 240
            bar_w = pw - 240 - 10
            bar_y = row_y + 22
            pygame.draw.rect(screen, (40, 50, 65), (bar_x, bar_y, bar_w, 6), border_radius=3)
            # Slider position
            ratio = (val - item['min']) / max(0.001, item['max'] - item['min'])
            fill_w = int(bar_w * ratio)
            color = (80, 200, 120) if ratio < 0.5 else (200, 200, 60) if ratio < 0.8 else (220, 100, 80)
            pygame.draw.rect(screen, color, (bar_x, bar_y, fill_w, 6), border_radius=3)

            # [-] and [+] buttons
            pygame.draw.rect(screen, (60, 70, 90), (minus_x, btn_y, btn_w, btn_h), border_radius=3)
            pygame.draw.rect(screen, (60, 70, 90), (plus_x, btn_y, btn_w, btn_h), border_radius=3)
            pygame.draw.rect(screen, (100, 120, 150), (minus_x, btn_y, btn_w, btn_h), 1, border_radius=3)
            pygame.draw.rect(screen, (100, 120, 150), (plus_x, btn_y, btn_w, btn_h), 1, border_radius=3)

            minus_t = self.font.render(" -", True, (200, 100, 100))
            plus_t = self.font.render(" +", True, (100, 200, 100))
            screen.blit(minus_t, (minus_x + 4, btn_y + 2))
            screen.blit(plus_t, (plus_x + 4, btn_y + 2))

    def handle_click(self, pos, settings):
        """Click handling. Returns True if something was modified."""
        if not self.visible:
            return False
        mx, my = pos
        px, py = self.panel_x, self.panel_y
        pw = self.panel_w

        for i, item in enumerate(self.ITEMS):
            row_y = py + self.header_h + i * self.row_h
            btn_y = row_y + 3
            btn_w, btn_h = 28, 22
            minus_x = px + pw - 80
            plus_x = px + pw - 42

            # [-] button
            if minus_x <= mx <= minus_x + btn_w and btn_y <= my <= btn_y + btn_h:
                val = settings[item['key']]
                settings[item['key']] = max(item['min'], val - item['step'])
                return True
            # [+] button
            if plus_x <= mx <= plus_x + btn_w and btn_y <= my <= btn_y + btn_h:
                val = settings[item['key']]
                settings[item['key']] = min(item['max'], val + item['step'])
                return True
        return False

    def handle_mouse_move(self, pos):
        if not self.visible:
            return
        mx, my = pos
        self.hovered_item = -1
        for i in range(len(self.ITEMS)):
            row_y = self.panel_y + self.header_h + i * self.row_h
            if row_y <= my <= row_y + self.row_h and self.panel_x <= mx <= self.panel_x + self.panel_w:
                self.hovered_item = i
                break

    def is_click_inside(self, pos):
        """Whether clicked inside the menu."""
        mx, my = pos
        return (self.panel_x <= mx <= self.panel_x + self.panel_w and
                self.panel_y <= my <= self.panel_y + self.panel_h)


# --- Game ---
class Game:
    def __init__(self):
        pygame.init()
        info = pygame.display.Info()
        # Fullscreen mode
        self.screen_w = info.current_w
        self.screen_h = info.current_h
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h), pygame.FULLSCREEN | pygame.SCALED)
        pygame.display.set_caption("Cell Evolution - Primordial Soup Ecosystem")
        self.clock = pygame.time.Clock()

        # World larger than screen
        self.world_w = self.screen_w * 2
        self.world_h = self.screen_h * 2

        self.world = World(self.world_w, self.world_h)
        self.renderer = Renderer(self.screen, self.world_w, self.world_h)
        self.settings_menu = SettingsMenu(self.screen_w, self.screen_h)

        self.paused = False
        self.sim_speed = 1
        self.running = True
        self.headless = False
        self.headless_render_interval = 1000  # Render every N ticks in headless mode
        self.headless_tps = 0  # Ticks per second (performance measurement)
        self._headless_tick_count = 0
        self._headless_last_time = 0

        # Camera to center
        self.renderer.cam_x = (self.world_w - self.screen_w) / 2
        self.renderer.cam_y = (self.world_h - self.screen_h) / 2

    def run(self):
        import time
        while self.running:
            self._handle_events()

            if self.headless:
                # HEADLESS MODE: max speed, no rendering
                if self._headless_last_time == 0:
                    self._headless_last_time = time.time()
                    self._headless_tick_count = 0

                # Run simulation without FPS limit
                batch = 50  # 50 ticks at once between event polling
                for _ in range(batch):
                    self.world.update()
                self._headless_tick_count += batch

                # TPS measurement
                now = time.time()
                elapsed = now - self._headless_last_time
                if elapsed >= 1.0:
                    self.headless_tps = int(self._headless_tick_count / elapsed)
                    self._headless_tick_count = 0
                    self._headless_last_time = now

                # Render only every N ticks
                if self.world.tick % self.headless_render_interval < batch:
                    self.renderer.draw(self.world, False, self.sim_speed)
                    # Headless indicator on screen
                    hud_surf = pygame.Surface((self.screen_w, 40), pygame.SRCALPHA)
                    hud_surf.fill((20, 0, 0, 200))
                    self.screen.blit(hud_surf, (0, 35))
                    hl_text = self.renderer.font_medium.render(
                        f"HEADLESS MODE  |  {self.headless_tps:,} tick/s  |  "
                        f"Tick: {self.world.tick:,}  |  "
                        f"Render: every {self.headless_render_interval}th tick  |  "
                        f"H: back  |  +/-: interval",
                        True, (255, 100, 100))
                    self.screen.blit(hl_text, (10, 41))
                    self.settings_menu.draw(self.screen, self.world.settings)
                    pygame.display.flip()
                self.clock.tick(0)  # No FPS limit!
            else:
                # NORMAL MODE
                if not self.paused:
                    for _ in range(self.sim_speed):
                        self.world.update()
                    # Action camera update (after simulation step)
                    if self.renderer.action_cam:
                        self.renderer.update_action_cam(self.world)
                self.renderer.draw(self.world, self.paused, self.sim_speed)
                self.settings_menu.draw(self.screen, self.world.settings)
                pygame.display.flip()
                self.clock.tick(FPS)
        pygame.quit()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_w:
                    self.sim_speed = min(20, self.sim_speed + 1)
                elif event.key == pygame.K_s:
                    if event.mod & pygame.KMOD_CTRL:
                        # SAVE
                        savepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cell_save.json")
                        try:
                            count = self.world.save_to_json(savepath)
                            self.renderer.flash_message = f"Saved! {count} cells -> cell_save.json"
                            self.renderer.flash_timer = 180
                        except Exception as e:
                            self.renderer.flash_message = f"Save error: {e}"
                            self.renderer.flash_timer = 240
                    else:
                        self.sim_speed = max(1, self.sim_speed - 1)
                elif event.key == pygame.K_l:
                    if event.mod & pygame.KMOD_CTRL:
                        # LOAD
                        savepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cell_save.json")
                        if os.path.exists(savepath):
                            try:
                                count = self.world.load_from_json(savepath)
                                self.renderer.selected_cell = None
                                self.renderer.flash_message = f"Loaded! {count} cells <- cell_save.json"
                                self.renderer.flash_timer = 180
                            except Exception as e:
                                self.renderer.flash_message = f"Load error: {e}"
                                self.renderer.flash_timer = 240
                        else:
                            self.renderer.flash_message = "No save file!"
                            self.renderer.flash_timer = 180
                elif event.key == pygame.K_f:
                    for _ in range(3):
                        self.world._spawn_food_cluster()
                elif event.key == pygame.K_1:
                    # Herbivore spawn at screen center
                    wx, wy = self.renderer.screen_to_world(self.screen_w // 2, self.screen_h // 2)
                    for _ in range(5):
                        self.world.spawn_herbivore(
                            wx + random.gauss(0, 30),
                            wy + random.gauss(0, 30))
                elif event.key == pygame.K_2:
                    # Predator spawn at screen center
                    wx, wy = self.renderer.screen_to_world(self.screen_w // 2, self.screen_h // 2)
                    for _ in range(3):
                        self.world.spawn_predator(
                            wx + random.gauss(0, 30),
                            wy + random.gauss(0, 30))
                elif event.key == pygame.K_3:
                    # Omnivore spawn at screen center
                    wx, wy = self.renderer.screen_to_world(self.screen_w // 2, self.screen_h // 2)
                    for _ in range(4):
                        self.world.spawn_omnivore(
                            wx + random.gauss(0, 30),
                            wy + random.gauss(0, 30))
                elif event.key == pygame.K_r:
                    self.world = World(self.world_w, self.world_h)
                    self.renderer.selected_cell = None
                elif event.key == pygame.K_TAB:
                    self.renderer.show_info = not self.renderer.show_info
                elif event.key == pygame.K_h:
                    self.headless = not self.headless
                    if self.headless:
                        self._headless_last_time = 0
                        self._headless_tick_count = 0
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS or event.key == pygame.K_KP_PLUS:
                    if self.headless:
                        self.headless_render_interval = min(10000, self.headless_render_interval + 500)
                elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                    if self.headless:
                        self.headless_render_interval = max(100, self.headless_render_interval - 500)
                elif event.key == pygame.K_c:
                    # Toggle action camera
                    self.renderer.action_cam = not self.renderer.action_cam
                    if self.renderer.action_cam:
                        self.renderer._action_saved_zoom = self.renderer.zoom
                        self.renderer.action_timer = 0  # Find event immediately
                        self.renderer.flash_message = "Action Camera ON"
                        self.renderer.flash_timer = 120
                    else:
                        self.renderer.zoom = self.renderer._action_saved_zoom
                        self.renderer.action_label = ""
                        self.renderer.flash_message = "Action Camera OFF"
                        self.renderer.flash_timer = 120
                elif event.key == pygame.K_m:
                    self.settings_menu.toggle()
                    if self.settings_menu.visible:
                        self.paused = True  # Pause when menu opens
            elif event.type == pygame.MOUSEMOTION:
                self.settings_menu.handle_mouse_move(event.pos)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # If menu is open and clicked inside
                    if self.settings_menu.visible:
                        if self.settings_menu.is_click_inside(event.pos):
                            self.settings_menu.handle_click(event.pos, self.world.settings)
                            # Sync global constants
                            global MUTATION_RATE, MUTATION_STRENGTH
                            MUTATION_RATE = self.world.settings['mutation_rate']
                            MUTATION_STRENGTH = self.world.settings['mutation_strength']
                            continue
                        else:
                            self.settings_menu.visible = False  # Click outside: close
                            continue
                    self.renderer.handle_click(event.pos, self.world)
                elif event.button == 4:  # Scroll up - zoom in
                    self.renderer.zoom = min(3.0, self.renderer.zoom * 1.1)
                elif event.button == 5:  # Scroll down - zoom out
                    self.renderer.zoom = max(0.3, self.renderer.zoom / 1.1)

        # Camera movement with arrows + mouse at edges (disabled during action cam)
        if self.renderer.action_cam:
            # Camera bounds still apply
            self.renderer.cam_x = max(0, min(self.world_w - self.screen_w / self.renderer.zoom,
                                              self.renderer.cam_x))
            self.renderer.cam_y = max(0, min(self.world_h - self.screen_h / self.renderer.zoom,
                                              self.renderer.cam_y))
            return
        keys = pygame.key.get_pressed()
        cam_speed = 12 / self.renderer.zoom
        if keys[pygame.K_LEFT]:
            self.renderer.cam_x -= cam_speed
        if keys[pygame.K_RIGHT]:
            self.renderer.cam_x += cam_speed
        if keys[pygame.K_UP]:
            self.renderer.cam_y -= cam_speed
        if keys[pygame.K_DOWN]:
            self.renderer.cam_y += cam_speed

        # Mouse edge scrolling
        mx, my = pygame.mouse.get_pos()
        edge = 20
        edge_speed = 6 / self.renderer.zoom
        if mx < edge:
            self.renderer.cam_x -= edge_speed
        elif mx > self.screen_w - edge:
            self.renderer.cam_x += edge_speed
        if my < edge:
            self.renderer.cam_y -= edge_speed
        elif my > self.screen_h - edge:
            self.renderer.cam_y += edge_speed

        # Camera bounds
        self.renderer.cam_x = max(0, min(self.world_w - self.screen_w / self.renderer.zoom,
                                          self.renderer.cam_x))
        self.renderer.cam_y = max(0, min(self.world_h - self.screen_h / self.renderer.zoom,
                                          self.renderer.cam_y))


def main():
    game = Game()
    game.run()


if __name__ == "__main__":
    main()
