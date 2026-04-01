"""
CelleR 2 — NEAT-Powered Cell Evolution Simulator
==================================================
Cells evolve neural networks (NEAT) to decide behaviour.
No hardcoded AI — everything is emergent.
"""

import pygame
import numpy as np
import math
import random
import json
import time
import os
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════

FPS = 60

# World
WORLD_MARGIN = 40  # soft boundary push

# Food
FOOD_COLOR = (50, 200, 50)
FOOD_RADIUS = 3
FOOD_ENERGY = 28
MAX_FOOD = 500
INITIAL_FOOD_OASES = 25
FOOD_PER_OASIS = 15
OASIS_RADIUS = 140
OASIS_REGROW_TICKS = 25  # ticks between regrow attempts per oasis
FOOD_MIN_ENERGY = 0.5

# Cells
INITIAL_CELLS = 60
MIN_POPULATION = 10
EXTINCTION_RESPAWN = 60
REPRODUCTION_COST_RATIO = 0.35
MUTATION_RATE = 0.15
MUTATION_STRENGTH = 0.12

# Eating
EAT_SPEED = 1.5
EAT_RANGE = 15.0
CORPSE_ENERGY_RATIO = 0.7
PLANT_DIGEST_HERB = 1.0
PLANT_DIGEST_OMNI = 0.55
PLANT_DIGEST_CARN = 0.1
MEAT_DIGEST_HERB = 0.1
MEAT_DIGEST_OMNI = 0.7
MEAT_DIGEST_CARN = 1.0

# Combat
ATTACK_RANGE = 15.0
ATTACK_COOLDOWN = 30  # ticks
HORN_DAMAGE_MULT = 0.5
STUN_DURATION = 20

# Overcrowding
OVERCROWDING_RADIUS = 50
OVERCROWDING_THRESHOLD = 5
OVERCROWDING_PENALTY = 0.03

# Pheromones
PHEROMONE_DECAY = 0.995
PHEROMONE_GRID_SIZE = 40

# Spatial grid
SPATIAL_GRID_SIZE = 60

# Shelters / obstacles
NUM_SHELTERS = 10
SHELTER_RADIUS = 35
NUM_OBSTACLES = 8
OBSTACLE_MIN_SIZE = 30
OBSTACLE_MAX_SIZE = 80

# Disease
DISEASE_CHECK_INTERVAL = 200
DISEASE_OUTBREAK_THRESHOLD = 0.38
DISEASE_OUTBREAK_CHANCE = 0.30
DISEASE_SPREAD_RANGE = 80
DISEASE_SPREAD_CHANCE = 0.04
DISEASE_DURATION = 250
DISEASE_IMMUNITY = 800
DISEASE_ENERGY_DRAIN = 0.06
DISEASE_SPEED_PENALTY = 0.7

# Water / thirst
NUM_WATER_SOURCES = 3
WATER_RADIUS = 55
WATER_MIN_DIST_FROM_FOOD = 500
THIRST_MAX = 100.0
THIRST_DRAIN = 0.018
THIRST_DRINK_RATE = 3.5
DEHYDRATION_DMG = 0.08

# Trail
TRAIL_INTERVAL = 3
TRAIL_MAX_LEN = 80
TRAIL_ALPHA_MAX = 50

# NEAT defaults
NEAT_COMPAT_THRESHOLD = 3.0
NEAT_C1 = 1.0  # excess gene coefficient
NEAT_C2 = 1.0  # disjoint gene coefficient
NEAT_C3 = 0.4  # weight difference coefficient
NEAT_WEIGHT_MUTATE_RATE = 0.80
NEAT_WEIGHT_RESET_RATE = 0.10
NEAT_ADD_CONN_RATE = 0.05
NEAT_ADD_NODE_RATE = 0.03
NEAT_TOGGLE_RATE = 0.02
NEAT_WEIGHT_PERTURB_STD = 0.3
NEAT_STAGNATION_LIMIT = 500  # ticks
NEAT_SPECIES_UPDATE_INTERVAL = 300  # ticks

# Brain energy cost (bloat prevention)
BRAIN_NODE_COST = 0.005
BRAIN_CONN_COST = 0.001

# Cilia limits
MAX_CILIA = 6

# Inputs / Outputs count
NUM_INPUTS = 24   # includes 2 memory_in
NUM_OUTPUTS = 18  # 6 cilia_power + 6 cilia_steer + attack + eat + mate + vocalize + mem_out0 + mem_out1

# Input indices
IN_ENERGY = 0
IN_HP = 1
IN_THIRST = 2
IN_SPEED = 3
IN_FOOD_DIST = 4
IN_FOOD_SIN = 5
IN_FOOD_COS = 6
IN_THREAT_DIST = 7
IN_THREAT_SIN = 8
IN_THREAT_COS = 9
IN_THREAT_SIZE = 10
IN_PREY_DIST = 11
IN_PREY_SIN = 12
IN_PREY_COS = 13
IN_MATE_DIST = 14
IN_MATE_SIN = 15
IN_MATE_COS = 16
IN_FOOD_SMELL_X = 17
IN_FOOD_SMELL_Y = 18
IN_DANGER_SMELL_X = 19
IN_DANGER_SMELL_Y = 20
IN_DENSITY = 21
IN_MEM_0 = 22
IN_MEM_1 = 23

# Output indices — cilia first, then actions
OUT_CILIA_POWER_0 = 0   # cilia 0 power (-1..+1)
OUT_CILIA_POWER_1 = 1
OUT_CILIA_POWER_2 = 2
OUT_CILIA_POWER_3 = 3
OUT_CILIA_POWER_4 = 4
OUT_CILIA_POWER_5 = 5
OUT_CILIA_STEER_0 = 6   # cilia 0 angle offset (-1..+1 → ±flex°)
OUT_CILIA_STEER_1 = 7
OUT_CILIA_STEER_2 = 8
OUT_CILIA_STEER_3 = 9
OUT_CILIA_STEER_4 = 10
OUT_CILIA_STEER_5 = 11
OUT_ATTACK = 12
OUT_EAT = 13
OUT_MATE = 14
OUT_VOCALIZE = 15
OUT_MEM_0 = 16
OUT_MEM_1 = 17

# Body gene indices
BG_SIZE = 0
BG_NUM_CILIA = 1
BG_CILIA_SPREAD = 2
BG_CILIA_POWER = 3
BG_CILIA_FLEX = 4
BG_DIET = 5
BG_HUE = 6
BG_METABOLISM = 7
BG_VISION_RANGE = 8
BG_HEARING_RANGE = 9
BG_HORN_SIZE = 10
BG_MOUTH_SIZE = 11
BG_NUM_HORNS = 12
BG_HORN_SPREAD = 13
BG_TURN_RATE = 14
NUM_BODY_GENES = 15

# Body gene ranges: (min, max)
BODY_GENE_RANGES = [
    (3.0, 25.0),     # size
    (1.0, 6.0),      # num_cilia (1-6)
    (0.0, 1.0),      # cilia_spread (0=torpedo, 1=jellyfish)
    (0.2, 3.0),      # cilia_power (force multiplier)
    (0.0, 30.0),     # cilia_flex (max steer angle in degrees)
    (0.0, 1.0),      # diet: 0=herbivore, 1=carnivore
    (0.0, 360.0),    # hue
    (0.2, 1.0),      # metabolism (lower=more efficient)
    (15.0, 350.0),   # vision_range
    (20.0, 400.0),   # hearing_range
    (0.0, 5.0),      # horn_size
    (0.5, 2.5),      # mouth_size
    (0.0, 3.0),      # num_horns (0-3, rounded)
    (0.0, 1.0),      # horn_spread (0=front, 1=radial)
    (0.02, 0.15),    # turn_rate (max rad/tick)
]

BODY_GENE_NAMES = [
    "size", "num_cilia", "cilia_spread", "cilia_power", "cilia_flex",
    "diet", "hue", "metabolism", "vision_range", "hearing_range", "horn_size",
    "mouth_size", "num_horns", "horn_spread", "turn_rate",
]

INPUT_NAMES = [
    "energy", "hp", "thirst", "speed",
    "food_d", "food_sin", "food_cos",
    "threat_d", "threat_sin", "threat_cos", "threat_sz",
    "prey_d", "prey_sin", "prey_cos",
    "mate_d", "mate_sin", "mate_cos",
    "smell_fx", "smell_fy", "smell_dx", "smell_dy",
    "density", "mem_in0", "mem_in1",
]

OUTPUT_NAMES = [
    "cil0_pw", "cil1_pw", "cil2_pw", "cil3_pw", "cil4_pw", "cil5_pw",
    "cil0_st", "cil1_st", "cil2_st", "cil3_st", "cil4_st", "cil5_st",
    "attack", "eat", "mate", "vocalize", "mem_out0", "mem_out1",
]


# ═══════════════════════════════════════════════════════════════
#  BODY GENOME
# ═══════════════════════════════════════════════════════════════

class BodyGenome:
    """15 physical genes defining the cell's body capabilities."""

    __slots__ = ('genes', '_cilia_cache', '_horn_cache')

    def __init__(self, genes=None):
        if genes is not None:
            self.genes = np.array(genes, dtype=np.float32)
        else:
            self.genes = np.array([
                random.uniform(lo, hi) for lo, hi in BODY_GENE_RANGES
            ], dtype=np.float32)
        self._cilia_cache = None  # cached cilia base angles
        self._horn_cache = None   # cached horn base angles

    # --- Property helpers ---
    @property
    def size(self):
        return float(self.genes[BG_SIZE])

    @property
    def num_cilia(self):
        return max(1, min(MAX_CILIA, int(self.genes[BG_NUM_CILIA])))

    @property
    def cilia_flex_rad(self):
        """Max cilia steer angle in radians."""
        return float(math.radians(self.genes[BG_CILIA_FLEX]))

    @property
    def diet(self):
        return float(self.genes[BG_DIET])

    @property
    def vision_range(self):
        return float(self.genes[BG_VISION_RANGE])

    @property
    def hearing_range(self):
        return float(self.genes[BG_HEARING_RANGE])

    @property
    def horn_size(self):
        return float(self.genes[BG_HORN_SIZE])

    @property
    def mouth_size(self):
        return float(self.genes[BG_MOUTH_SIZE])

    @property
    def num_horns(self):
        return max(0, min(3, round(self.genes[BG_NUM_HORNS])))

    @property
    def horn_spread(self):
        return float(self.genes[BG_HORN_SPREAD])

    @property
    def turn_rate(self):
        return float(self.genes[BG_TURN_RATE])

    @property
    def metabolism(self):
        return float(self.genes[BG_METABOLISM])

    @property
    def hue(self):
        return float(self.genes[BG_HUE])

    @property
    def mouth_cone_half(self):
        """Half-angle of eating cone in radians."""
        deg = 55.0 + self.genes[BG_MOUTH_SIZE] * 10.0
        return math.radians(deg)

    @property
    def eat_speed_mult(self):
        """Eating speed multiplier from mouth size."""
        return float(0.7 + self.genes[BG_MOUTH_SIZE] * 0.25)

    @property
    def max_speed(self):
        """Theoretical max speed if all cilia at full power in same direction."""
        n = self.num_cilia
        cilia_p = self.genes[BG_CILIA_POWER]
        size = self.genes[BG_SIZE]
        return float(n * cilia_p * 0.5 / (size * 0.08 + 1.0))

    @property
    def max_hp(self):
        return float(self.genes[BG_SIZE] * 4 + self.genes[BG_HORN_SIZE] * self.num_horns * 0.7)

    @property
    def repro_threshold(self):
        return float(35 + self.genes[BG_SIZE] * 1.5)

    @property
    def attack_power(self):
        base = self.genes[BG_SIZE] * 0.3 + self.genes[BG_HORN_SIZE] * self.num_horns * 1.2
        diet = self.genes[BG_DIET]
        # Carnivore synergy: horns are offensive weapons
        base *= (1.0 + diet * 0.3)
        return float(base)

    @property
    def horn_defense(self):
        """Passive defense damage dealt back to attacker."""
        base = self.genes[BG_HORN_SIZE] * self.num_horns * HORN_DAMAGE_MULT
        diet = self.genes[BG_DIET]
        # Herbivore synergy: horns are defensive
        base *= (1.0 + (1.0 - diet) * 0.5)
        return float(base)

    @property
    def energy_cap(self):
        return self.repro_threshold * 2.5

    def base_energy_drain(self):
        """Per-tick energy drain from body upkeep."""
        s = self.genes
        drain = 0.0
        drain += s[BG_SIZE] * 0.0008
        drain += s[BG_NUM_CILIA] * 0.0006
        drain += s[BG_CILIA_POWER] * 0.001
        drain += s[BG_CILIA_FLEX] * 0.00005  # flexible joints cost a tiny bit
        drain += s[BG_VISION_RANGE] * 0.00004
        drain += s[BG_HEARING_RANGE] * 0.00003
        drain += s[BG_HORN_SIZE] * 0.001
        drain += s[BG_MOUTH_SIZE] * 0.0003
        drain += max(0, round(s[BG_NUM_HORNS])) * 0.0004  # per horn
        drain += s[BG_TURN_RATE] * 0.002  # agile joints cost energy
        drain *= s[BG_METABOLISM]
        return float(drain)

    def plant_efficiency(self):
        d = self.diet
        # Continuous curve: herbivore(0)=1.0, omnivore(0.5)=0.61, carnivore(1)=0.0
        return float(max(0.0, (1.0 - d) ** 0.7))

    def _plant_efficiency_old(self):
        d = self.diet
        if d < 0.3:
            return PLANT_DIGEST_HERB
        elif d > 0.7:
            return PLANT_DIGEST_CARN
        return PLANT_DIGEST_OMNI

    def meat_efficiency(self):
        d = self.diet
        # Continuous curve: herbivore(0)=0.0, omnivore(0.5)=0.61, carnivore(1)=1.0
        return float(max(0.0, d ** 0.7))

    def diet_label(self):
        d = self.diet
        if d < 0.3:
            return "herbivore"
        elif d > 0.7:
            return "carnivore"
        return "omnivore"

    def cilia_positions(self):
        """Return list of base angles (relative to cell heading) where cilia attach.
        Cached until genes change."""
        if self._cilia_cache is not None:
            return self._cilia_cache
        n = self.num_cilia
        spread = self.genes[BG_CILIA_SPREAD]
        if spread < 0.3:
            # Torpedo mode: cilia clustered at the back
            angles = [math.pi + (i - (n - 1) / 2) * 0.4 for i in range(n)]
        elif spread > 0.7:
            # Jellyfish mode: cilia evenly spread around
            angles = [2 * math.pi * i / n for i in range(n)]
        else:
            # Mixed: some back, some sides
            angles = []
            for i in range(n):
                if i % 2 == 0:
                    angles.append(math.pi + (i - (n - 1) / 2) * 0.3)
                else:
                    angles.append(math.pi * 0.5 + i * 0.5)
        self._cilia_cache = angles
        return angles

    def horn_positions(self):
        """Return list of base angles (relative to cell heading) where horns attach.
        Cached until genes change."""
        if self._horn_cache is not None:
            return self._horn_cache
        n = self.num_horns
        if n == 0:
            self._horn_cache = []
            return []
        spread = self.genes[BG_HORN_SPREAD]
        if spread < 0.3:
            # Frontal: horns clustered at front
            if n == 1:
                angles = [0.0]
            else:
                angles = [(i - (n - 1) / 2) * 0.35 for i in range(n)]
        elif spread > 0.7:
            # Radial: evenly around body
            angles = [2 * math.pi * i / n for i in range(n)]
        else:
            # Mixed: some front, some sides
            angles = []
            for i in range(n):
                if i == 0:
                    angles.append(0.0)  # one always at front
                else:
                    angles.append(math.pi * 0.4 * i * (1 if i % 2 == 0 else -1))
        self._horn_cache = angles
        return angles

    def color(self):
        """HSV-based color from hue gene, diet affects saturation."""
        h = self.hue / 360.0
        s = 0.6 + self.diet * 0.4  # carnivores more saturated
        v = 0.8
        c = pygame.Color(0)
        c.hsva = (h * 360, s * 100, v * 100, 100)
        return (c.r, c.g, c.b)

    def mutate(self):
        """In-place mutation."""
        for i in range(NUM_BODY_GENES):
            if random.random() < MUTATION_RATE:
                lo, hi = BODY_GENE_RANGES[i]
                self.genes[i] += random.gauss(0, MUTATION_STRENGTH * (hi - lo))
                self.genes[i] = np.clip(self.genes[i], lo, hi)
        self._cilia_cache = None
        self._horn_cache = None

    @staticmethod
    def crossover(parent_a, parent_b):
        """Uniform crossover of body genes."""
        child_genes = np.empty(NUM_BODY_GENES, dtype=np.float32)
        for i in range(NUM_BODY_GENES):
            child_genes[i] = parent_a.genes[i] if random.random() < 0.5 else parent_b.genes[i]
        child = BodyGenome(child_genes)
        child.mutate()
        return child

    def to_dict(self):
        return {"genes": self.genes.tolist()}

    @staticmethod
    def from_dict(d):
        return BodyGenome(d["genes"])


# ═══════════════════════════════════════════════════════════════
#  NEAT INNOVATION TRACKER (global singleton)
# ═══════════════════════════════════════════════════════════════

class InnovationTracker:
    """Tracks structural mutations to assign consistent innovation numbers."""

    def __init__(self):
        self.next_innovation = 0
        self.next_node_id = NUM_INPUTS + NUM_OUTPUTS  # 0..23=input, 24..33=output
        # (in_node, out_node) → innovation number
        self.innovations = {}

    def get_innovation(self, in_node, out_node):
        key = (in_node, out_node)
        if key in self.innovations:
            return self.innovations[key]
        innov = self.next_innovation
        self.next_innovation += 1
        self.innovations[key] = innov
        return innov

    def new_node_id(self):
        nid = self.next_node_id
        self.next_node_id += 1
        return nid

    def to_dict(self):
        return {
            "next_innovation": self.next_innovation,
            "next_node_id": self.next_node_id,
            "innovations": {f"{k[0]},{k[1]}": v for k, v in self.innovations.items()},
        }

    @staticmethod
    def from_dict(d):
        t = InnovationTracker()
        t.next_innovation = d["next_innovation"]
        t.next_node_id = d["next_node_id"]
        t.innovations = {}
        for k, v in d["innovations"].items():
            parts = k.split(",")
            t.innovations[(int(parts[0]), int(parts[1]))] = v
        return t


# Global tracker — shared by all genomes
_innovation_tracker = InnovationTracker()


def get_innovation_tracker():
    return _innovation_tracker


def set_innovation_tracker(tracker):
    global _innovation_tracker
    _innovation_tracker = tracker


# ═══════════════════════════════════════════════════════════════
#  NEAT GENOME
# ═══════════════════════════════════════════════════════════════

# Node types
NODE_INPUT = 0
NODE_OUTPUT = 1
NODE_HIDDEN = 2

# Activation functions
ACT_TANH = 0
ACT_SIGMOID = 1
ACT_RELU = 2
ACT_IDENTITY = 3

_ACT_FUNCS = {
    ACT_TANH: np.tanh,
    ACT_SIGMOID: lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10))),
    ACT_RELU: lambda x: np.maximum(0, x),
    ACT_IDENTITY: lambda x: x,
}


class NeatGenome:
    """NEAT genome: nodes + connections with innovation numbers."""

    __slots__ = ('nodes', 'connections', '_network_cache')

    def __init__(self):
        # nodes: {node_id: (node_type, activation_fn)}
        self.nodes = {}
        # connections: {innovation_num: (in_node, out_node, weight, enabled)}
        self.connections = {}
        # Compiled network cache (invalidated on topology change)
        self._network_cache = None

    def invalidate_cache(self):
        self._network_cache = None

    @staticmethod
    def create_minimal(tracker=None):
        """Create a minimal genome with all inputs connected to all outputs (random weights)."""
        if tracker is None:
            tracker = get_innovation_tracker()
        g = NeatGenome()
        # Input nodes: 0..NUM_INPUTS-1
        for i in range(NUM_INPUTS):
            g.nodes[i] = (NODE_INPUT, ACT_IDENTITY)
        # Output nodes: NUM_INPUTS..NUM_INPUTS+NUM_OUTPUTS-1
        for i in range(NUM_OUTPUTS):
            nid = NUM_INPUTS + i
            g.nodes[nid] = (NODE_OUTPUT, ACT_TANH)
        # Connect each input to each output with small random weight
        for i_node in range(NUM_INPUTS):
            for o_idx in range(NUM_OUTPUTS):
                o_node = NUM_INPUTS + o_idx
                innov = tracker.get_innovation(i_node, o_node)
                w = random.gauss(0, 0.5)
                g.connections[innov] = (i_node, o_node, w, True)
        return g

    @staticmethod
    def create_sparse(tracker=None, connection_prob=0.15):
        """Create a sparse genome with seeded survival connections.
        Seeded connections give cells basic food-seeking + eating behaviour
        so the initial population can survive long enough to evolve."""
        if tracker is None:
            tracker = get_innovation_tracker()
        g = NeatGenome()
        for i in range(NUM_INPUTS):
            g.nodes[i] = (NODE_INPUT, ACT_IDENTITY)
        for i in range(NUM_OUTPUTS):
            nid = NUM_INPUTS + i
            g.nodes[nid] = (NODE_OUTPUT, ACT_TANH)

        # --- Seeded survival connections (with noise) ---
        # Key insight: invertált distance → 0 = nothing, 1 = close.
        # So we need BASELINE movement from smell/energy, and BOOST from proximity.
        seed_conns = [
            # Baseline forward drive: energy → cilia power (have energy = move)
            (IN_ENERGY, OUT_CILIA_POWER_0, random.uniform(0.8, 1.5)),
            (IN_ENERGY, OUT_CILIA_POWER_1, random.uniform(0.8, 1.5)),
            # Smell-based navigation: food smell → steer cilia (stronger for directional eating)
            (IN_FOOD_SMELL_X, OUT_CILIA_STEER_0, random.uniform(0.8, 1.5)),
            (IN_FOOD_SMELL_Y, OUT_CILIA_STEER_0, random.uniform(0.5, 1.0)),
            (IN_FOOD_SMELL_X, OUT_CILIA_STEER_1, random.uniform(0.4, 0.9)),
            (IN_FOOD_SMELL_X, OUT_CILIA_POWER_0, random.uniform(0.3, 0.8)),
            # Visual food approach: boost + steer when food visible
            (IN_FOOD_DIST, OUT_CILIA_POWER_0, random.uniform(0.5, 1.2)),
            (IN_FOOD_DIST, OUT_CILIA_POWER_1, random.uniform(0.3, 0.8)),
            (IN_FOOD_SIN, OUT_CILIA_STEER_0, random.uniform(0.8, 1.5)),
            (IN_FOOD_SIN, OUT_CILIA_STEER_1, random.uniform(0.5, 1.0)),
            (IN_FOOD_COS, OUT_CILIA_POWER_0, random.uniform(0.2, 0.6)),
            # Eat always on (bias) — let network learn to NOT eat
            (IN_FOOD_DIST, OUT_EAT, random.uniform(1.5, 2.5)),
            (IN_ENERGY, OUT_EAT, random.uniform(0.3, 0.6)),
            # Threat → stop cilia (flee by stopping)
            (IN_THREAT_DIST, OUT_CILIA_POWER_0, random.uniform(-1.5, -0.8)),
            (IN_THREAT_DIST, OUT_CILIA_POWER_1, random.uniform(-1.2, -0.5)),
            # Energy → mate (when well-fed, seek mates)
            (IN_ENERGY, OUT_MATE, random.uniform(0.3, 0.8)),
            # Cilia 2-5 also get some base drive
            (IN_ENERGY, OUT_CILIA_POWER_2, random.uniform(0.5, 1.0)),
            (IN_FOOD_DIST, OUT_CILIA_POWER_2, random.uniform(0.3, 0.6)),
            (IN_FOOD_SIN, OUT_CILIA_STEER_2, random.uniform(0.3, 0.8)),
            # Mate scent → steer toward mates
            (IN_DANGER_SMELL_X, OUT_CILIA_STEER_0, random.uniform(0.2, 0.5)),
            (IN_DANGER_SMELL_Y, OUT_CILIA_POWER_0, random.uniform(0.2, 0.5)),
            # Vocalize when wanting to mate
            (IN_ENERGY, OUT_VOCALIZE, random.uniform(0.2, 0.5)),
        ]
        for in_n, out_idx, w in seed_conns:
            o_node = NUM_INPUTS + out_idx
            innov = tracker.get_innovation(in_n, o_node)
            g.connections[innov] = (in_n, o_node, w, True)

        # --- Random sparse connections on top ---
        for i_node in range(NUM_INPUTS):
            for o_idx in range(NUM_OUTPUTS):
                if random.random() < connection_prob:
                    o_node = NUM_INPUTS + o_idx
                    innov = tracker.get_innovation(i_node, o_node)
                    if innov not in g.connections:  # don't overwrite seeds
                        w = random.gauss(0, 0.5)
                        g.connections[innov] = (i_node, o_node, w, True)

        # Ensure every output has at least one connection
        for o_idx in range(NUM_OUTPUTS):
            o_node = NUM_INPUTS + o_idx
            has_input = any(
                conn[1] == o_node and conn[3]
                for conn in g.connections.values()
            )
            if not has_input:
                i_node = random.randint(0, NUM_INPUTS - 1)
                innov = tracker.get_innovation(i_node, o_node)
                w = random.gauss(0, 0.5)
                g.connections[innov] = (i_node, o_node, w, True)
        return g

    # --- Mutation ---

    def mutate(self, tracker=None):
        """Apply mutations. Called once per reproduction event."""
        if tracker is None:
            tracker = get_innovation_tracker()

        # Weight mutations
        for innov in list(self.connections):
            in_n, out_n, w, enabled = self.connections[innov]
            if not enabled:
                continue
            if random.random() < NEAT_WEIGHT_MUTATE_RATE:
                if random.random() < NEAT_WEIGHT_RESET_RATE:
                    w = random.gauss(0, 0.5)
                else:
                    w += random.gauss(0, NEAT_WEIGHT_PERTURB_STD)
                w = max(-8.0, min(8.0, w))
                self.connections[innov] = (in_n, out_n, w, enabled)

        # Add connection
        if random.random() < NEAT_ADD_CONN_RATE:
            self._mutate_add_connection(tracker)

        # Add node
        if random.random() < NEAT_ADD_NODE_RATE:
            self._mutate_add_node(tracker)

        # Toggle connections
        for innov in list(self.connections):
            if random.random() < NEAT_TOGGLE_RATE:
                in_n, out_n, w, enabled = self.connections[innov]
                self.connections[innov] = (in_n, out_n, w, not enabled)
                self.invalidate_cache()

    def _mutate_add_connection(self, tracker):
        """Add a new connection between two unconnected nodes (feedforward only)."""
        node_ids = list(self.nodes.keys())
        # Try up to 20 times to find a valid new connection
        for _ in range(20):
            a = random.choice(node_ids)
            b = random.choice(node_ids)
            if a == b:
                continue
            # No input→input or output→output
            a_type = self.nodes[a][0]
            b_type = self.nodes[b][0]
            if a_type == NODE_OUTPUT or b_type == NODE_INPUT:
                # Swap if needed, or skip
                if b_type == NODE_OUTPUT and a_type != NODE_OUTPUT:
                    a, b = a, b  # a→b is fine (input/hidden → output)
                elif a_type == NODE_INPUT and b_type != NODE_INPUT:
                    a, b = a, b  # input → hidden/output is fine
                else:
                    continue
            # Check not already connected
            existing = any(
                conn[0] == a and conn[1] == b
                for conn in self.connections.values()
            )
            if existing:
                continue
            # Check no cycle would be created (feedforward only)
            if self._would_create_cycle(a, b):
                continue
            innov = tracker.get_innovation(a, b)
            w = random.gauss(0, 0.5)
            self.connections[innov] = (a, b, w, True)
            self.invalidate_cache()
            return

    def _would_create_cycle(self, from_node, to_node):
        """Check if adding from_node→to_node would create a cycle."""
        # BFS from to_node: if we can reach from_node, it's a cycle
        visited = set()
        stack = [to_node]
        while stack:
            current = stack.pop()
            if current == from_node:
                return True
            if current in visited:
                continue
            visited.add(current)
            for conn in self.connections.values():
                if conn[0] == current and conn[3]:  # enabled
                    stack.append(conn[1])
        return False

    def _mutate_add_node(self, tracker):
        """Split an existing connection with a new hidden node."""
        enabled_conns = [
            (innov, conn) for innov, conn in self.connections.items() if conn[3]
        ]
        if not enabled_conns:
            return
        innov_old, (in_n, out_n, w, _) = random.choice(enabled_conns)
        # Disable old connection
        self.connections[innov_old] = (in_n, out_n, w, False)
        # New hidden node
        new_id = tracker.new_node_id()
        self.nodes[new_id] = (NODE_HIDDEN, ACT_TANH)
        # in_n → new (weight 1.0)
        innov1 = tracker.get_innovation(in_n, new_id)
        self.connections[innov1] = (in_n, new_id, 1.0, True)
        # new → out_n (old weight)
        innov2 = tracker.get_innovation(new_id, out_n)
        self.connections[innov2] = (new_id, out_n, w, True)
        self.invalidate_cache()

    # --- Crossover ---

    @staticmethod
    def crossover(parent_a, parent_b, a_is_fitter=True):
        """NEAT crossover. Matching genes: random pick. Disjoint/excess: from fitter parent."""
        child = NeatGenome()

        # All node IDs from fitter parent (+ any matching from other)
        fitter = parent_a if a_is_fitter else parent_b
        other = parent_b if a_is_fitter else parent_a

        all_innovs = set(fitter.connections.keys()) | set(other.connections.keys())

        for innov in all_innovs:
            in_fitter = innov in fitter.connections
            in_other = innov in other.connections

            if in_fitter and in_other:
                # Matching gene — random pick
                conn = fitter.connections[innov] if random.random() < 0.5 else other.connections[innov]
            elif in_fitter:
                # Disjoint/excess from fitter
                conn = fitter.connections[innov]
            else:
                # Disjoint/excess from weaker — skip (standard NEAT)
                continue

            child.connections[innov] = conn

        # Collect all referenced nodes
        for conn in child.connections.values():
            in_n, out_n = conn[0], conn[1]
            if in_n not in child.nodes:
                if in_n in fitter.nodes:
                    child.nodes[in_n] = fitter.nodes[in_n]
                elif in_n in other.nodes:
                    child.nodes[in_n] = other.nodes[in_n]
            if out_n not in child.nodes:
                if out_n in fitter.nodes:
                    child.nodes[out_n] = fitter.nodes[out_n]
                elif out_n in other.nodes:
                    child.nodes[out_n] = other.nodes[out_n]

        # Ensure all input/output nodes exist
        for i in range(NUM_INPUTS):
            if i not in child.nodes:
                child.nodes[i] = (NODE_INPUT, ACT_IDENTITY)
        for i in range(NUM_OUTPUTS):
            nid = NUM_INPUTS + i
            if nid not in child.nodes:
                child.nodes[nid] = (NODE_OUTPUT, ACT_TANH)

        return child

    # --- Compatibility distance ---

    def distance(self, other):
        """NEAT compatibility distance for speciation."""
        innovs_a = set(self.connections.keys())
        innovs_b = set(other.connections.keys())

        if not innovs_a and not innovs_b:
            return 0.0

        matching = innovs_a & innovs_b
        all_innovs = innovs_a | innovs_b

        if not all_innovs:
            return 0.0

        max_a = max(innovs_a) if innovs_a else 0
        max_b = max(innovs_b) if innovs_b else 0
        max_common = min(max_a, max_b)

        excess = 0
        disjoint = 0
        weight_diff = 0.0

        for innov in all_innovs:
            in_a = innov in innovs_a
            in_b = innov in innovs_b
            if in_a and in_b:
                # Matching — accumulate weight difference
                weight_diff += abs(self.connections[innov][2] - other.connections[innov][2])
            elif innov > max_common:
                excess += 1
            else:
                disjoint += 1

        n_matching = len(matching) if matching else 1
        N = max(len(innovs_a), len(innovs_b))
        if N < 20:
            N = 1  # small genomes: don't normalize

        return (NEAT_C1 * excess / N +
                NEAT_C2 * disjoint / N +
                NEAT_C3 * weight_diff / n_matching)

    # --- Brain cost ---

    def brain_cost(self):
        """Energy per tick for maintaining this brain."""
        n_hidden = sum(1 for ntype, _ in self.nodes.values() if ntype == NODE_HIDDEN)
        n_enabled = sum(1 for conn in self.connections.values() if conn[3])
        return n_hidden * BRAIN_NODE_COST + n_enabled * BRAIN_CONN_COST

    # --- Network info ---

    def num_hidden(self):
        return sum(1 for ntype, _ in self.nodes.values() if ntype == NODE_HIDDEN)

    def num_enabled_connections(self):
        return sum(1 for conn in self.connections.values() if conn[3])

    # --- Serialization ---

    def to_dict(self):
        return {
            "nodes": {str(k): [v[0], v[1]] for k, v in self.nodes.items()},
            "connections": {
                str(k): [v[0], v[1], v[2], v[3]]
                for k, v in self.connections.items()
            },
        }

    @staticmethod
    def from_dict(d):
        g = NeatGenome()
        g.nodes = {int(k): (v[0], v[1]) for k, v in d["nodes"].items()}
        g.connections = {
            int(k): (v[0], v[1], v[2], bool(v[3]))
            for k, v in d["connections"].items()
        }
        return g


# ═══════════════════════════════════════════════════════════════
#  NEAT NETWORK (compiled, numpy-optimized)
# ═══════════════════════════════════════════════════════════════

class NeatNetwork:
    """Compiled feedforward network from a NeatGenome.
    Uses Kahn's algorithm for topological ordering.
    Results cached until topology changes."""

    __slots__ = ('layers', 'node_order', 'node_activations',
                 'conn_weights', 'conn_in', 'conn_out',
                 'input_ids', 'output_ids', 'valid')

    def __init__(self):
        self.valid = False

    @staticmethod
    def compile(genome):
        """Compile a NeatGenome into an efficient NeatNetwork."""
        net = NeatNetwork()

        # Collect enabled connections
        enabled = [(innov, conn) for innov, conn in genome.connections.items() if conn[3]]

        # Build adjacency and in-degree for Kahn's algorithm
        adj = defaultdict(list)      # node → [downstream nodes]
        in_degree = defaultdict(int)
        active_nodes = set()

        for _, (in_n, out_n, w, _) in enabled:
            adj[in_n].append(out_n)
            in_degree[out_n] += 1
            active_nodes.add(in_n)
            active_nodes.add(out_n)

        # Ensure all input/output nodes are present
        for i in range(NUM_INPUTS):
            active_nodes.add(i)
        for i in range(NUM_OUTPUTS):
            active_nodes.add(NUM_INPUTS + i)

        # Initialize queue with nodes having no incoming edges (inputs)
        queue = []
        for node in active_nodes:
            if in_degree.get(node, 0) == 0:
                queue.append(node)
        queue.sort()  # deterministic ordering

        # Kahn's topological sort
        topo_order = []
        while queue:
            node = queue.pop(0)
            topo_order.append(node)
            for neighbor in adj.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # If not all nodes sorted → cycle detected (shouldn't happen in feedforward)
        if len(topo_order) < len(active_nodes):
            # Fallback: just use active nodes in id order
            topo_order = sorted(active_nodes)

        # Assign layer numbers for visualization
        node_layer = {}
        for node in topo_order:
            if genome.nodes.get(node, (NODE_HIDDEN,))[0] == NODE_INPUT:
                node_layer[node] = 0
            else:
                max_parent = 0
                for _, (in_n, out_n, w, en) in enabled:
                    if out_n == node and en:
                        max_parent = max(max_parent, node_layer.get(in_n, 0) + 1)
                node_layer[node] = max_parent

        # Build compiled arrays
        # Node evaluation order (skip inputs — they're set directly)
        eval_order = [n for n in topo_order if genome.nodes.get(n, (NODE_HIDDEN,))[0] != NODE_INPUT]

        # Connection arrays
        n_conns = len(enabled)
        conn_in = np.zeros(n_conns, dtype=np.int32)
        conn_out = np.zeros(n_conns, dtype=np.int32)
        conn_weights = np.zeros(n_conns, dtype=np.float32)
        for i, (_, (in_n, out_n, w, _)) in enumerate(enabled):
            conn_in[i] = in_n
            conn_out[i] = out_n
            conn_weights[i] = w

        # Node activation functions
        node_acts = {}
        for nid, (ntype, act) in genome.nodes.items():
            node_acts[nid] = act

        net.node_order = eval_order
        net.node_activations = node_acts
        net.conn_in = conn_in
        net.conn_out = conn_out
        net.conn_weights = conn_weights
        net.input_ids = list(range(NUM_INPUTS))
        net.output_ids = [NUM_INPUTS + i for i in range(NUM_OUTPUTS)]
        net.layers = node_layer
        net.valid = True

        return net

    def activate(self, inputs):
        """Run forward pass. inputs: array of NUM_INPUTS floats.
        Returns: array of NUM_OUTPUTS floats (tanh-activated)."""
        # Node values dict
        values = {}

        # Set input values
        for i in range(NUM_INPUTS):
            values[i] = inputs[i]

        # Initialize all eval nodes to 0
        for node in self.node_order:
            values[node] = 0.0

        # Accumulate weighted inputs from connections
        for i in range(len(self.conn_weights)):
            src = int(self.conn_in[i])
            dst = int(self.conn_out[i])
            w = self.conn_weights[i]
            src_val = values.get(src, 0.0)
            values[dst] = values.get(dst, 0.0) + src_val * w

        # Apply activations in topological order
        for node in self.node_order:
            act = self.node_activations.get(node, ACT_TANH)
            v = values[node]
            if act == ACT_TANH:
                values[node] = math.tanh(v)
            elif act == ACT_SIGMOID:
                values[node] = 1.0 / (1.0 + math.exp(-max(-10, min(10, v))))
            elif act == ACT_RELU:
                values[node] = max(0.0, v)
            # ACT_IDENTITY: no change

        # Collect outputs
        outputs = np.zeros(NUM_OUTPUTS, dtype=np.float32)
        for i in range(NUM_OUTPUTS):
            outputs[i] = values.get(self.output_ids[i], 0.0)

        return outputs


# ═══════════════════════════════════════════════════════════════
#  PHEROMONE MAP (numpy, from CelleR 1)
# ═══════════════════════════════════════════════════════════════

class PheromoneMap:
    """Chemical signals on the map — numpy-backed."""
    TRAIL = 0
    DANGER = 1
    FOOD_HERE = 2
    AMBUSH = 3
    MATE = 4
    PREY_SCENT = 5
    CORPSE_SCENT = 6
    NUM_TYPES = 7

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid_size = PHEROMONE_GRID_SIZE
        self.cols = width // self.grid_size + 1
        self.rows = height // self.grid_size + 1
        self.data = np.zeros((self.rows, self.cols, self.NUM_TYPES), dtype=np.float32)

    def deposit(self, x, y, ptype, intensity=1.0):
        col = int(x / self.grid_size)
        row = int(y / self.grid_size)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.data[row, col, ptype] = min(self.data[row, col, ptype] + intensity, 5.0)

    def deposit_cloud(self, x, y, ptype, intensity=1.0, radius=3):
        col = int(x / self.grid_size)
        row = int(y / self.grid_size)
        r1 = max(0, row - radius)
        r2 = min(self.rows, row + radius + 1)
        c1 = max(0, col - radius)
        c2 = min(self.cols, col + radius + 1)
        inv_radius = 1.0 / (radius + 1)
        radius_sq = radius * radius
        for r in range(r1, r2):
            dr = r - row
            for c in range(c1, c2):
                dc = c - col
                if dr * dr + dc * dc <= radius_sq:
                    falloff = max(0.1, 1.0 - (abs(dr) + abs(dc)) * 0.5 * inv_radius)
                    self.data[r, c, ptype] = min(self.data[r, c, ptype] + intensity * falloff, 5.0)

    def read(self, x, y, ptype, radius=1):
        col = int(x / self.grid_size)
        row = int(y / self.grid_size)
        r1 = max(0, row - radius)
        r2 = min(self.rows, row + radius + 1)
        c1 = max(0, col - radius)
        c2 = min(self.cols, col + radius + 1)
        return float(self.data[r1:r2, c1:c2, ptype].sum())

    def read_gradient(self, x, y, ptype):
        col = int(x / self.grid_size)
        row = int(y / self.grid_size)
        r1 = max(0, row - 2)
        r2 = min(self.rows, row + 3)
        c1 = max(0, col - 2)
        c2 = min(self.cols, col + 3)
        patch = self.data[r1:r2, c1:c2, ptype]
        if patch.sum() < 0.05:
            return 0.0, 0.0
        dr_start = r1 - row
        dc_start = c1 - col
        rows_p, cols_p = patch.shape
        gx = 0.0
        gy = 0.0
        for ri in range(rows_p):
            for ci in range(cols_p):
                v = patch[ri, ci]
                if v > 0:
                    gx += (dc_start + ci) * v
                    gy += (dr_start + ri) * v
        return gx, gy

    _DECAY_COMPENSATED = np.float32(PHEROMONE_DECAY ** 3)

    def decay(self):
        self.data *= self._DECAY_COMPENSATED
        self.data[self.data < 0.05] = 0


# ═══════════════════════════════════════════════════════════════
#  SPATIAL GRID
# ═══════════════════════════════════════════════════════════════

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
        results = []
        cs = self.cell_size
        min_gx = int((x - radius) // cs)
        max_gx = int((x + radius) // cs)
        min_gy = int((y - radius) // cs)
        max_gy = int((y + radius) // cs)
        grid = self.grid
        _extend = results.extend
        for gx in range(min_gx, max_gx + 1):
            for gy in range(min_gy, max_gy + 1):
                bucket = grid.get((gx, gy))
                if bucket:
                    _extend(bucket)
        return results


# ═══════════════════════════════════════════════════════════════
#  INPUT BUILDER
# ═══════════════════════════════════════════════════════════════

class InputBuilder:
    """Builds the 24-float input vector for a cell's neural network."""

    @staticmethod
    def build(cell, world):
        """Returns numpy array of NUM_INPUTS floats, all normalized."""
        inp = np.zeros(NUM_INPUTS, dtype=np.float32)
        bg = cell.body_genome

        # Self state
        inp[IN_ENERGY] = min(cell.energy / bg.repro_threshold, 2.0) / 2.0
        inp[IN_HP] = cell.hp / bg.max_hp if bg.max_hp > 0 else 1.0
        inp[IN_THIRST] = cell.thirst / THIRST_MAX
        inp[IN_SPEED] = min(cell.speed / bg.max_speed, 1.0) if bg.max_speed > 0 else 0.0

        # Vision-based perceptions (inverted distance: 1=close, 0=far/nothing)
        vis = bg.vision_range
        cx, cy = cell.x, cell.y
        ca = cell.angle

        # Find nearest food, threat, prey, mate
        nearby = world.spatial.query(cx, cy, vis)

        best_food_d = vis + 1
        best_food = None
        best_threat_d = vis + 1
        best_threat = None
        best_prey_d = vis + 1
        best_prey = None
        best_mate_d = vis + 1
        best_mate = None

        my_size = bg.size
        my_diet = bg.diet

        for obj in nearby:
            if obj is cell:
                continue

            # Food items
            if isinstance(obj, Food):
                dx = obj.x - cx
                dy = obj.y - cy
                d = math.sqrt(dx * dx + dy * dy)
                if d < best_food_d:
                    best_food_d = d
                    best_food = obj
                continue

            # Carcasses count as food for meat-eaters
            if isinstance(obj, Carcass):
                if my_diet > 0.2:  # can eat meat
                    dx = obj.x - cx
                    dy = obj.y - cy
                    d = math.sqrt(dx * dx + dy * dy)
                    if d < best_food_d:
                        best_food_d = d
                        best_food = obj
                continue

            if not isinstance(obj, Cell):
                continue

            other = obj
            if not other.alive:
                continue

            dx = other.x - cx
            dy = other.y - cy
            d = math.sqrt(dx * dx + dy * dy)
            if d > vis:
                continue

            other_size = other.body_genome.size
            other_diet = other.body_genome.diet

            # Classify: threat, prey, or mate?
            size_ratio = other_size / max(my_size, 0.1)

            # Threat: bigger carnivore/omnivore
            if other_diet > 0.3 and size_ratio > 0.8:
                if d < best_threat_d:
                    best_threat_d = d
                    best_threat = other
            # Prey: smaller herbivore/omnivore (if I'm carnivore-ish)
            elif my_diet > 0.3 and size_ratio < 1.2:
                if d < best_prey_d:
                    best_prey_d = d
                    best_prey = other
            # Mate: compatible species, similar size
            if d < best_mate_d and cell.can_mate_with(other):
                best_mate_d = d
                best_mate = other

        # Fill food input (inverted distance + angle)
        if best_food is not None and best_food_d <= vis:
            inp[IN_FOOD_DIST] = 1.0 - best_food_d / vis
            angle_to = math.atan2(best_food.y - cy, best_food.x - cx) - ca
            inp[IN_FOOD_SIN] = math.sin(angle_to)
            inp[IN_FOOD_COS] = math.cos(angle_to)

        # Fill threat input
        if best_threat is not None and best_threat_d <= vis:
            inp[IN_THREAT_DIST] = 1.0 - best_threat_d / vis
            angle_to = math.atan2(best_threat.y - cy, best_threat.x - cx) - ca
            inp[IN_THREAT_SIN] = math.sin(angle_to)
            inp[IN_THREAT_COS] = math.cos(angle_to)
            inp[IN_THREAT_SIZE] = min(best_threat.body_genome.size / max(my_size, 0.1), 2.0) / 2.0

        # Fill prey input
        if best_prey is not None and best_prey_d <= vis:
            inp[IN_PREY_DIST] = 1.0 - best_prey_d / vis
            angle_to = math.atan2(best_prey.y - cy, best_prey.x - cx) - ca
            inp[IN_PREY_SIN] = math.sin(angle_to)
            inp[IN_PREY_COS] = math.cos(angle_to)

        # Fill mate input
        if best_mate is not None and best_mate_d <= vis:
            inp[IN_MATE_DIST] = 1.0 - best_mate_d / vis
            angle_to = math.atan2(best_mate.y - cy, best_mate.x - cx) - ca
            inp[IN_MATE_SIN] = math.sin(angle_to)
            inp[IN_MATE_COS] = math.cos(angle_to)

        # Smell gradients
        pm = world.pheromone_map
        if my_diet < 0.5:
            # Herbivore: smell plant food
            gx, gy = pm.read_gradient(cx, cy, PheromoneMap.FOOD_HERE)
        else:
            # Carnivore: smell prey scent
            gx, gy = pm.read_gradient(cx, cy, PheromoneMap.PREY_SCENT)
        mag = math.sqrt(gx * gx + gy * gy) + 0.001
        inp[IN_FOOD_SMELL_X] = max(-1.0, min(1.0, gx / mag))
        inp[IN_FOOD_SMELL_Y] = max(-1.0, min(1.0, gy / mag))

        # Secondary smell: danger gradient OR mate pheromone (context-dependent)
        # Well-fed cells smell for mates; hungry/threatened cells smell danger
        if cell.energy > bg.repro_threshold * 0.6:
            # Smell mate pheromone
            dgx, dgy = pm.read_gradient(cx, cy, PheromoneMap.MATE)
        else:
            dgx, dgy = pm.read_gradient(cx, cy, PheromoneMap.DANGER)
        dmag = math.sqrt(dgx * dgx + dgy * dgy) + 0.001
        inp[IN_DANGER_SMELL_X] = max(-1.0, min(1.0, dgx / dmag))
        inp[IN_DANGER_SMELL_Y] = max(-1.0, min(1.0, dgy / dmag))

        # Nearby density
        density = sum(1 for o in nearby if isinstance(o, Cell) and o is not cell and o.alive)
        inp[IN_DENSITY] = min(density / 10.0, 1.0)

        # Memory from previous tick
        inp[IN_MEM_0] = cell.memory_state[0]
        inp[IN_MEM_1] = cell.memory_state[1]

        return inp


# ═══════════════════════════════════════════════════════════════
#  OUTPUT INTERPRETER
# ═══════════════════════════════════════════════════════════════

class OutputInterpreter:
    """Converts 18-float neural output into cilia forces + actions."""

    @staticmethod
    def apply(cell, outputs):
        """Apply neural network outputs to cell state.
        Cilia outputs → per-cilium power and steer stored on cell.
        Actions → intent flags."""
        bg = cell.body_genome
        n_cilia = bg.num_cilia
        flex = bg.cilia_flex_rad  # max steer in radians

        # Per-cilium power (0..1, positive only) and steer (-flex..+flex)
        for i in range(n_cilia):
            cell.cilia_powers[i] = (outputs[OUT_CILIA_POWER_0 + i] + 1.0) * 0.5  # tanh→0..1
            cell.cilia_steers[i] = outputs[OUT_CILIA_STEER_0 + i] * flex
        # Zero out unused slots
        for i in range(n_cilia, MAX_CILIA):
            cell.cilia_powers[i] = 0.0
            cell.cilia_steers[i] = 0.0

        # Action intents
        cell.attack_intent = outputs[OUT_ATTACK] > 0
        cell.eat_intent = outputs[OUT_EAT] > 0
        cell.mate_intent = outputs[OUT_MATE] > 0
        cell.vocalize_intent = outputs[OUT_VOCALIZE] > 0

        # Memory output → stored for next tick
        cell.memory_state[0] = max(-1.0, min(1.0, outputs[OUT_MEM_0]))
        cell.memory_state[1] = max(-1.0, min(1.0, outputs[OUT_MEM_1]))


# ═══════════════════════════════════════════════════════════════
#  FOOD & CARCASS
# ═══════════════════════════════════════════════════════════════

class Food:
    __slots__ = ('x', 'y', 'energy', 'oasis_id')

    def __init__(self, x, y, energy=FOOD_ENERGY, oasis_id=-1):
        self.x = x
        self.y = y
        self.energy = energy
        self.oasis_id = oasis_id


class Carcass:
    __slots__ = ('x', 'y', 'energy', 'age', 'size')

    def __init__(self, x, y, energy, size=5.0):
        self.x = x
        self.y = y
        self.energy = energy
        self.age = 0
        self.size = size


# ═══════════════════════════════════════════════════════════════
#  CELL
# ═══════════════════════════════════════════════════════════════

_next_cell_id = 0

def _new_cell_id():
    global _next_cell_id
    _next_cell_id += 1
    return _next_cell_id


class Cell:
    """A living cell with a body genome and a NEAT brain."""

    __slots__ = (
        'id', 'x', 'y', 'vx', 'vy', 'angle', 'speed',
        'body_genome', 'neat_genome', 'network',
        'energy', 'hp', 'thirst', 'alive', 'age',
        'memory_state', 'fitness',
        'cilia_powers', 'cilia_steers',
        'attack_intent', 'eat_intent', 'mate_intent',
        'vocalize_intent',
        'attack_cooldown', 'stun_timer',
        'juvenile', 'juvenile_timer',
        'sick', 'sick_timer', 'immunity_timer',
        'trail', 'trail_timer', 'cilia_phase', 'damaged_flash',
        'parent_id', 'generation', 'children_count',
        'kills', 'food_eaten',
        'species_id',
    )

    def __init__(self, x, y, body_genome=None, neat_genome=None, parent_id=None, generation=0):
        self.id = _new_cell_id()
        self.x = x
        self.y = y
        self.angle = random.uniform(0, 2 * math.pi)
        self.vx = 0.0
        self.vy = 0.0
        self.speed = 0.0

        self.body_genome = body_genome or BodyGenome()
        self.neat_genome = neat_genome or NeatGenome.create_sparse()
        self.network = NeatNetwork.compile(self.neat_genome)

        bg = self.body_genome
        self.energy = bg.repro_threshold * 0.9
        self.hp = bg.max_hp
        self.thirst = THIRST_MAX * 0.5
        self.alive = True
        self.age = 0

        # Neural memory (persistent between ticks)
        self.memory_state = [0.0, 0.0]

        # Fitness (for species ranking, NOT reproduction trigger)
        self.fitness = 0.0

        # Cilia state (set by OutputInterpreter each tick)
        self.cilia_powers = [0.0] * MAX_CILIA   # per-cilium thrust (-1..+1)
        self.cilia_steers = [0.0] * MAX_CILIA   # per-cilium angle offset (radians)

        # Action intents
        self.attack_intent = False
        self.eat_intent = False
        self.mate_intent = False
        self.vocalize_intent = False

        # Combat
        self.attack_cooldown = 0
        self.stun_timer = 0

        # Juvenile (can't reproduce or fight)
        self.juvenile = True
        self.juvenile_timer = int(60 + bg.size * 3)

        # Disease
        self.sick = False
        self.sick_timer = 0
        self.immunity_timer = 0

        # Visual
        self.trail = []
        self.trail_timer = 0
        self.cilia_phase = random.uniform(0, 2 * math.pi)
        self.damaged_flash = 0

        # Lineage
        self.parent_id = parent_id
        self.generation = generation
        self.children_count = 0
        self.kills = 0
        self.food_eaten = 0

        # Species (assigned by NeatPopulation)
        self.species_id = -1

    def can_mate_with(self, other):
        """Check if two cells can reproduce."""
        if not self.alive or not other.alive:
            return False
        if self.juvenile or other.juvenile:
            return False
        if not self.mate_intent or not other.mate_intent:
            return False
        bg_a = self.body_genome
        bg_b = other.body_genome
        # At least one parent must be above repro threshold, other needs 60%
        a_ready = self.energy >= bg_a.repro_threshold
        b_ready = other.energy >= bg_b.repro_threshold
        if not a_ready and not b_ready:
            return False
        if self.energy < bg_a.repro_threshold * 0.6 or other.energy < bg_b.repro_threshold * 0.6:
            return False
        # NEAT compatibility check
        dist = self.neat_genome.distance(other.neat_genome)
        return dist < NEAT_COMPAT_THRESHOLD * 1.5  # slightly looser for mating vs speciation

    def get_compiled_network(self):
        """Get or compile the neural network."""
        if self.neat_genome._network_cache is not None:
            return self.neat_genome._network_cache
        net = NeatNetwork.compile(self.neat_genome)
        self.neat_genome._network_cache = net
        self.network = net
        return net

    def update_physics(self, world_w, world_h, obstacles):
        """Update position based on neural outputs."""
        bg = self.body_genome

        # Age & visuals
        self.age += 1
        self.cilia_phase += self.speed * 0.3 + 0.05  # faster movement = faster cilia
        if self.damaged_flash > 0:
            self.damaged_flash -= 1

        # Juvenile countdown
        if self.juvenile:
            self.juvenile_timer -= 1
            if self.juvenile_timer <= 0:
                self.juvenile = False

        # Disease
        if self.sick:
            self.sick_timer -= 1
            if self.sick_timer <= 0:
                self.sick = False
                self.immunity_timer = DISEASE_IMMUNITY
        if self.immunity_timer > 0:
            self.immunity_timer -= 1

        # Stun
        if self.stun_timer > 0:
            self.stun_timer -= 1
            self.vx *= 0.9
            self.vy *= 0.9
            self.speed = math.sqrt(self.vx * self.vx + self.vy * self.vy)
            self.x += self.vx
            self.y += self.vy
            self._clamp_position(world_w, world_h)
            return

        # Attack cooldown
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1

        # ── CILIA-BASED PHYSICS ──
        # Each cilium is a thrust vector. The cell's acceleration is the
        # sum of all cilia vectors. Turning happens naturally when cilia
        # on one side push harder or are steered differently.
        n_cilia = bg.num_cilia
        base_angles = bg.cilia_positions()
        cilia_force = bg.genes[BG_CILIA_POWER] * 0.3

        ax = 0.0
        ay = 0.0
        torque = 0.0  # net rotational force

        for i in range(n_cilia):
            power = self.cilia_powers[i]         # 0..1 (positive only)
            steer = self.cilia_steers[i]         # radians offset

            # Cilium absolute angle = cell heading + base position + steer offset
            base_a = base_angles[i]
            cil_angle = self.angle + base_a + steer

            # Force vector: cilium pushes OPPOSITE to its direction
            # (cilium points backward → pushes cell forward)
            push_angle = cil_angle + math.pi  # opposite direction
            force = power * cilia_force
            ax += math.cos(push_angle) * force
            ay += math.sin(push_angle) * force

            # Torque: off-center cilia create rotation
            # Cross product of position vector × force vector
            # Position on cell surface (relative to center)
            px = math.cos(self.angle + base_a)
            py = math.sin(self.angle + base_a)
            fx = math.cos(push_angle) * force
            fy = math.sin(push_angle) * force
            torque += px * fy - py * fx  # 2D cross product

        # Apply torque as angular acceleration, clamped by turn_rate
        inertia = bg.size * bg.size * 0.01 + 0.1
        angular_delta = torque / inertia
        max_turn = bg.turn_rate
        angular_delta = max(-max_turn, min(max_turn, angular_delta))
        self.angle += angular_delta
        self.angle %= (2 * math.pi)

        # Speed limit
        max_spd = bg.max_speed
        if self.sick:
            max_spd *= DISEASE_SPEED_PENALTY

        self.vx += ax
        self.vy += ay

        # Friction
        self.vx *= 0.92
        self.vy *= 0.92

        # Speed limit
        self.speed = math.sqrt(self.vx * self.vx + self.vy * self.vy)
        if self.speed > max_spd:
            scale = max_spd / self.speed
            self.vx *= scale
            self.vy *= scale
            self.speed = max_spd

        # Position update
        self.x += self.vx
        self.y += self.vy

        # Obstacle collision
        r = bg.size
        for obs in obstacles:
            ox, oy, ow, oh = obs
            # Simple AABB collision
            closest_x = max(ox, min(self.x, ox + ow))
            closest_y = max(oy, min(self.y, oy + oh))
            dx = self.x - closest_x
            dy = self.y - closest_y
            d2 = dx * dx + dy * dy
            if d2 < r * r and d2 > 0:
                d = math.sqrt(d2)
                push = (r - d) / d
                self.x += dx * push
                self.y += dy * push
                self.vx *= 0.5
                self.vy *= 0.5

        self._clamp_position(world_w, world_h)

        # Trail
        self.trail_timer += 1
        if self.trail_timer >= TRAIL_INTERVAL:
            self.trail_timer = 0
            self.trail.append((self.x, self.y))
            if len(self.trail) > TRAIL_MAX_LEN:
                self.trail.pop(0)

    def _clamp_position(self, world_w, world_h):
        m = WORLD_MARGIN
        if self.x < m:
            self.x = m
            self.vx = abs(self.vx) * 0.5
        elif self.x > world_w - m:
            self.x = world_w - m
            self.vx = -abs(self.vx) * 0.5
        if self.y < m:
            self.y = m
            self.vy = abs(self.vy) * 0.5
        elif self.y > world_h - m:
            self.y = world_h - m
            self.vy = -abs(self.vy) * 0.5

    def take_damage(self, amount):
        self.hp -= amount
        self.damaged_flash = 6
        if self.hp <= 0:
            self.hp = 0
            self.alive = False

    def die(self):
        self.alive = False
        self.hp = 0


# ═══════════════════════════════════════════════════════════════
#  NEAT POPULATION (speciation & management)
# ═══════════════════════════════════════════════════════════════

class NeatSpecies:
    __slots__ = ('id', 'representative', 'members', 'avg_fitness',
                 'best_fitness', 'stagnation_counter', 'age')

    def __init__(self, species_id, representative):
        self.id = species_id
        self.representative = representative  # NeatGenome
        self.members = []  # list of Cell
        self.avg_fitness = 0.0
        self.best_fitness = 0.0
        self.stagnation_counter = 0
        self.age = 0


class NeatPopulation:
    """Manages NEAT species for the real-time simulation."""

    def __init__(self):
        self.species = {}  # species_id → NeatSpecies
        self.next_species_id = 0

    def classify(self, cells):
        """Assign each cell to a species based on NEAT compatibility."""
        # Clear members
        for sp in self.species.values():
            sp.members = []

        for cell in cells:
            if not cell.alive:
                continue
            placed = False
            for sp in self.species.values():
                dist = cell.neat_genome.distance(sp.representative)
                if dist < NEAT_COMPAT_THRESHOLD:
                    sp.members.append(cell)
                    cell.species_id = sp.id
                    placed = True
                    break
            if not placed:
                # New species
                sid = self.next_species_id
                self.next_species_id += 1
                new_sp = NeatSpecies(sid, cell.neat_genome)
                new_sp.members.append(cell)
                cell.species_id = sid
                self.species[sid] = new_sp

        # Remove empty species
        empty = [sid for sid, sp in self.species.items() if not sp.members]
        for sid in empty:
            del self.species[sid]

        # Update representatives and fitness stats
        for sp in self.species.values():
            sp.representative = random.choice(sp.members).neat_genome
            sp.age += 1

            # Calculate average fitness with shared fitness (dividing by species size)
            if sp.members:
                total = sum(c.fitness for c in sp.members)
                sp.avg_fitness = total / len(sp.members)
                best = max(c.fitness for c in sp.members)
                if best > sp.best_fitness:
                    sp.best_fitness = best
                    sp.stagnation_counter = 0
                else:
                    sp.stagnation_counter += NEAT_SPECIES_UPDATE_INTERVAL

    def get_stagnant_species(self):
        """Returns species that have stagnated (no improvement)."""
        return [sp for sp in self.species.values()
                if sp.stagnation_counter >= NEAT_STAGNATION_LIMIT]

    def generate_latin_name(self, species_id):
        """Generate a pseudo-Latin species name from the species ID."""
        prefixes = [
            "Micro", "Proto", "Neo", "Pseudo", "Para", "Meta",
            "Mega", "Ultra", "Hyper", "Poly", "Mono", "Tri",
            "Crypto", "Endo", "Exo", "Xeno", "Pyro", "Cryo",
        ]
        roots = [
            "cell", "morph", "plasm", "soma", "phage", "cyte",
            "zoa", "blast", "mere", "sect", "pod", "branchi",
            "derm", "gastr", "neur", "arth", "chond", "angi",
        ]
        suffixes = [
            "us", "is", "um", "ae", "oides", "ensis",
            "alis", "icus", "atus", "inus", "osus", "iformis",
        ]

        random.seed(species_id * 7919)  # deterministic per species
        name = random.choice(prefixes) + random.choice(roots) + random.choice(suffixes)
        random.seed()  # restore randomness
        return name


# ═══════════════════════════════════════════════════════════════
#  WORLD
# ═══════════════════════════════════════════════════════════════

class World:
    """Main simulation world."""

    def __init__(self, width, height, boot_camp=True):
        self.width = width
        self.height = height
        self.tick = 0

        # Core systems
        self.pheromone_map = PheromoneMap(width, height)
        self.spatial = SpatialGrid()
        self.neat_pop = NeatPopulation()

        # Entities
        self.cells = []
        self.foods = []
        self.carcasses = []

        # Oases (food spawn points)
        self.oases = []
        self._generate_oases()

        # Water sources
        self.water_sources = []
        self._generate_water_sources()

        # Obstacles (rocks)
        self.obstacles = []
        self._generate_obstacles()

        # Shelters
        self.shelters = []
        self._generate_shelters()

        # Stats
        self.pop_history = []  # (tick, herb_count, pred_count, omni_count)
        self.total_born = 0
        self.total_died = 0

        # Disease tracking
        self.disease_timer = 0

        # Auto-respawn control (disabled for main world — user manages population)
        self.auto_respawn = (boot_camp == 'arena')

        # Lineage tracking
        self.lineage = {}  # cell_id → {parent_id, generation, born_tick, species_name, ...}

        # Spawn initial cells (with boot camp for main world, not for arena sub-worlds)
        self._spawn_initial_cells(boot_camp=boot_camp)

        # Initial food spawn
        self._spawn_initial_food()

    def _generate_oases(self):
        for i in range(INITIAL_FOOD_OASES):
            ox = random.randint(100, self.width - 100)
            oy = random.randint(100, self.height - 100)
            self.oases.append((ox, oy, i))

    def _generate_water_sources(self):
        for _ in range(NUM_WATER_SOURCES):
            for attempt in range(50):
                wx = random.randint(100, self.width - 100)
                wy = random.randint(100, self.height - 100)
                # Check minimum distance from oases
                too_close = False
                for ox, oy, _ in self.oases:
                    if math.hypot(wx - ox, wy - oy) < WATER_MIN_DIST_FROM_FOOD:
                        too_close = True
                        break
                if not too_close:
                    self.water_sources.append((wx, wy))
                    break
            else:
                self.water_sources.append((wx, wy))  # fallback

    def _generate_obstacles(self):
        for _ in range(NUM_OBSTACLES):
            ox = random.randint(50, self.width - 100)
            oy = random.randint(50, self.height - 100)
            ow = random.randint(OBSTACLE_MIN_SIZE, OBSTACLE_MAX_SIZE)
            oh = random.randint(OBSTACLE_MIN_SIZE, OBSTACLE_MAX_SIZE)
            self.obstacles.append((ox, oy, ow, oh))

    def _generate_shelters(self):
        for _ in range(NUM_SHELTERS):
            sx = random.randint(50, self.width - 50)
            sy = random.randint(50, self.height - 50)
            self.shelters.append((sx, sy))

    def _spawn_initial_cells(self, boot_camp=True):
        """Spawn initial cells. boot_camp='arena' for arena sub-worlds, False for empty."""
        if boot_camp == 'arena':
            # Arena sub-world: spawn random untrained cells
            for _ in range(INITIAL_CELLS):
                x = random.randint(100, self.width - 100)
                y = random.randint(100, self.height - 100)
                cell = Cell(x, y)
                self.cells.append(cell)
                self._record_lineage(cell)
            return

        # Main world: start empty — user populates from F9 arena + genome library
        # (food still spawns at oases, world is ready to receive injected cells)

    def _spawn_initial_food(self):
        for ox, oy, oid in self.oases:
            for _ in range(FOOD_PER_OASIS):
                fx = ox + random.gauss(0, OASIS_RADIUS * 0.4)
                fy = oy + random.gauss(0, OASIS_RADIUS * 0.4)
                fx = max(10, min(self.width - 10, fx))
                fy = max(10, min(self.height - 10, fy))
                self.foods.append(Food(fx, fy, FOOD_ENERGY, oid))

    def _record_lineage(self, cell):
        self.lineage[cell.id] = {
            'parent_id': cell.parent_id,
            'generation': cell.generation,
            'born_tick': self.tick,
            'death_tick': None,
            'species_name': None,
            'children': [],
        }
        if cell.parent_id and cell.parent_id in self.lineage:
            self.lineage[cell.parent_id]['children'].append(cell.id)

    # ──────── TICK ────────

    def update(self):
        """One simulation tick."""
        self.tick += 1

        # Rebuild spatial grid
        self.spatial.clear()
        for cell in self.cells:
            if cell.alive:
                self.spatial.insert(cell, cell.x, cell.y)
        for food in self.foods:
            self.spatial.insert(food, food.x, food.y)
        for carcass in self.carcasses:
            self.spatial.insert(carcass, carcass.x, carcass.y)

        # Brain tick: build inputs → run network → interpret outputs
        throttle = self.tick % 2
        for cell in self.cells:
            if not cell.alive:
                continue
            # AI throttling: non-urgent cells think every 2nd tick
            if (cell.id + throttle) % 2 == 0 or cell.stun_timer > 0:
                if cell.age > 1:  # always think on first tick
                    pass  # skip this tick, keep old actions
                else:
                    self._think(cell)
            else:
                self._think(cell)

        # Physics, energy, combat, eating, reproduction
        for cell in self.cells:
            if not cell.alive:
                continue
            cell.update_physics(self.width, self.height, self.obstacles)
            self._process_energy(cell)
            self._process_thirst(cell)

        self._process_combat()
        self._process_eating()
        self._process_reproduction()

        # Pheromone deposits + decay
        if self.tick % 3 == 0:
            self.pheromone_map.decay()
        self._deposit_pheromones()

        # Food regrowth
        if self.tick % OASIS_REGROW_TICKS == 0:
            self._regrow_food()

        # Carcass aging
        self._age_carcasses()

        # Disease
        self.disease_timer += 1
        if self.disease_timer >= DISEASE_CHECK_INTERVAL:
            self.disease_timer = 0
            self._check_disease()
        self._spread_disease()

        # Remove dead cells, create carcasses
        self._process_deaths()

        # Population management (only if auto_respawn enabled — arena worlds)
        if self.auto_respawn:
            living = [c for c in self.cells if c.alive]
            if len(living) == 0:
                self._extinction_respawn()
            elif len(living) < MIN_POPULATION:
                self._spawn_random_cells(5)

        # Species classification (periodic)
        if self.tick % NEAT_SPECIES_UPDATE_INTERVAL == 0:
            living = [c for c in self.cells if c.alive]
            self.neat_pop.classify(living)
            # Handle stagnation
            for sp in self.neat_pop.get_stagnant_species():
                for cell in sp.members:
                    # Double mutation rate for stagnant species
                    cell.neat_genome.mutate()
                    cell.neat_genome.invalidate_cache()

        # Population history
        if self.tick % 10 == 0:
            self._record_population()

        # Clean dead cells list (keep last 500 for lineage)
        if len(self.cells) > 800:
            alive_cells = [c for c in self.cells if c.alive]
            dead_recent = [c for c in self.cells if not c.alive][-200:]
            self.cells = alive_cells + dead_recent

    def _think(self, cell):
        """Run neural network for one cell."""
        inputs = InputBuilder.build(cell, self)
        net = cell.get_compiled_network()
        outputs = net.activate(inputs)
        OutputInterpreter.apply(cell, outputs)

    def _process_energy(self, cell):
        """Energy drain per tick."""
        bg = cell.body_genome
        drain = bg.base_energy_drain()

        # Brain cost (bloat prevention)
        drain += cell.neat_genome.brain_cost()

        # High cilia exertion cost (proportional to total power output)
        total_power = sum(p for p in cell.cilia_powers[:cell.body_genome.num_cilia])
        drain += total_power * 0.004

        # Overcrowding
        nearby = self.spatial.query(cell.x, cell.y, OVERCROWDING_RADIUS)
        neighbors = sum(1 for o in nearby if isinstance(o, Cell) and o is not cell and o.alive)
        if neighbors > OVERCROWDING_THRESHOLD:
            drain += (neighbors - OVERCROWDING_THRESHOLD) * OVERCROWDING_PENALTY

        # Disease
        if cell.sick:
            drain += DISEASE_ENERGY_DRAIN

        cell.energy -= drain

        # Energy cap
        if cell.energy > bg.energy_cap:
            cell.energy = bg.energy_cap

        # Starvation
        if cell.energy <= 0:
            cell.energy = 0
            cell.take_damage(0.1)

        # Fitness accumulation
        cell.fitness += 0.001  # survival reward

    def _process_thirst(self, cell):
        """Thirst system."""
        cell.thirst -= THIRST_DRAIN

        # Check if near water
        for wx, wy in self.water_sources:
            d = math.hypot(cell.x - wx, cell.y - wy)
            if d < WATER_RADIUS:
                cell.thirst = min(THIRST_MAX, cell.thirst + THIRST_DRINK_RATE)
                break

        if cell.thirst <= 0:
            cell.thirst = 0
            cell.take_damage(DEHYDRATION_DMG)

    def _process_combat(self):
        """Handle attacks between cells."""
        for cell in self.cells:
            if not cell.alive or not cell.attack_intent:
                continue
            if cell.attack_cooldown > 0 or cell.juvenile:
                continue

            bg = cell.body_genome
            nearby = self.spatial.query(cell.x, cell.y, ATTACK_RANGE + bg.size)

            for other in nearby:
                if not isinstance(other, Cell) or other is cell or not other.alive:
                    continue
                dx = other.x - cell.x
                dy = other.y - cell.y
                d = math.sqrt(dx * dx + dy * dy)
                if d > ATTACK_RANGE + bg.size:
                    continue

                # Check angle (attack in facing direction, ~90° cone)
                angle_to = math.atan2(dy, dx)
                angle_diff = abs(((angle_to - cell.angle + math.pi) % (2 * math.pi)) - math.pi)
                if angle_diff > 0.8:
                    continue

                # Damage (attack_power already includes diet synergy)
                dmg = bg.attack_power
                # Defender horn passive defense (includes herbivore synergy)
                defender_horn_dmg = other.body_genome.horn_defense
                other.take_damage(dmg)
                cell.take_damage(defender_horn_dmg)

                # Stun
                other.stun_timer = STUN_DURATION

                cell.attack_cooldown = ATTACK_COOLDOWN
                cell.energy -= 0.002  # attack cost

                if not other.alive:
                    cell.kills += 1
                    cell.fitness += 2.0
                break  # one attack per tick

    def _process_eating(self):
        """Handle eating food and carcasses — directional (mouth cone only)."""
        for cell in self.cells:
            if not cell.alive or not cell.eat_intent:
                continue

            bg = cell.body_genome
            eat_range = EAT_RANGE * (0.8 + bg.mouth_size * 0.15) + bg.size
            mouth_half = bg.mouth_cone_half
            eat_spd = EAT_SPEED * bg.eat_speed_mult
            nearby = self.spatial.query(cell.x, cell.y, eat_range)

            for obj in nearby:
                if isinstance(obj, Food):
                    dx = obj.x - cell.x
                    dy = obj.y - cell.y
                    dist_sq = dx * dx + dy * dy
                    if dist_sq > eat_range * eat_range:
                        continue
                    # Skip cone check if food overlaps cell body (absorbed on contact)
                    if dist_sq > bg.size * bg.size:
                        # Mouth cone check: food must be in front of cell
                        angle_to = math.atan2(dy, dx)
                        diff = (angle_to - cell.angle + math.pi) % (2 * math.pi) - math.pi
                        if abs(diff) > mouth_half:
                            continue
                    # Eat plant
                    eff = bg.plant_efficiency()
                    amount = min(eat_spd, obj.energy)
                    gained = amount * eff
                    obj.energy -= amount
                    cell.energy += gained
                    cell.food_eaten += 1
                    cell.fitness += gained * 0.01
                    self.pheromone_map.deposit(cell.x, cell.y, PheromoneMap.FOOD_HERE, 0.5)
                    break

                elif isinstance(obj, Carcass):
                    dx = obj.x - cell.x
                    dy = obj.y - cell.y
                    dist_sq = dx * dx + dy * dy
                    if dist_sq > eat_range * eat_range:
                        continue
                    # Skip cone check if carcass overlaps cell body
                    if dist_sq > bg.size * bg.size:
                        # Mouth cone check
                        angle_to = math.atan2(dy, dx)
                        diff = (angle_to - cell.angle + math.pi) % (2 * math.pi) - math.pi
                        if abs(diff) > mouth_half:
                            continue
                    # Eat meat
                    eff = bg.meat_efficiency()
                    amount = min(eat_spd, obj.energy)
                    gained = amount * eff
                    obj.energy -= amount
                    cell.energy += gained
                    cell.food_eaten += 1
                    cell.fitness += gained * 0.01
                    break

        # Remove depleted food
        self.foods = [f for f in self.foods if f.energy >= FOOD_MIN_ENERGY]
        self.carcasses = [c for c in self.carcasses if c.energy >= FOOD_MIN_ENERGY]

    def _process_reproduction(self):
        """Handle mating and offspring."""
        new_cells = []
        mated_this_tick = set()

        for cell in self.cells:
            if not cell.alive or not cell.mate_intent or cell.juvenile:
                continue
            if cell.id in mated_this_tick:
                continue
            bg = cell.body_genome
            if cell.energy < bg.repro_threshold * 0.6:
                continue

            # Find mate
            nearby = self.spatial.query(cell.x, cell.y, bg.size + 30)
            for other in nearby:
                if not isinstance(other, Cell) or other is cell or not other.alive:
                    continue
                if other.id in mated_this_tick:
                    continue
                if not cell.can_mate_with(other):
                    continue

                dx = other.x - cell.x
                dy = other.y - cell.y
                d = math.sqrt(dx * dx + dy * dy)
                if d > bg.size + other.body_genome.size + 20:
                    continue

                # Reproduce!
                child = self._make_offspring(cell, other)
                new_cells.append(child)

                # Energy cost
                cell.energy *= (1 - REPRODUCTION_COST_RATIO)
                other.energy *= (1 - REPRODUCTION_COST_RATIO)

                cell.children_count += 1
                other.children_count += 1
                cell.fitness += 5.0
                other.fitness += 5.0

                mated_this_tick.add(cell.id)
                mated_this_tick.add(other.id)
                self.total_born += 1
                break

        for child in new_cells:
            self.cells.append(child)
            self._record_lineage(child)

    def _make_offspring(self, parent_a, parent_b):
        """Create a child cell from two parents."""
        # Body genome crossover
        child_body = BodyGenome.crossover(parent_a.body_genome, parent_b.body_genome)

        # NEAT genome crossover
        a_fitter = parent_a.fitness >= parent_b.fitness
        child_brain = NeatGenome.crossover(parent_a.neat_genome, parent_b.neat_genome, a_fitter)
        child_brain.mutate()

        # Spawn near parents
        mx = (parent_a.x + parent_b.x) / 2
        my = (parent_a.y + parent_b.y) / 2
        cx = mx + random.gauss(0, 15)
        cy = my + random.gauss(0, 15)
        cx = max(50, min(self.width - 50, cx))
        cy = max(50, min(self.height - 50, cy))

        child = Cell(cx, cy, child_body, child_brain,
                     parent_id=parent_a.id,
                     generation=max(parent_a.generation, parent_b.generation) + 1)
        return child

    def _deposit_pheromones(self):
        """Cells deposit pheromones based on their state."""
        pm = self.pheromone_map
        for cell in self.cells:
            if not cell.alive:
                continue
            # Trail
            pm.deposit(cell.x, cell.y, PheromoneMap.TRAIL, 0.3)
            # Prey scent (herbivores leave scent for predators to track)
            if cell.body_genome.diet < 0.5:
                pm.deposit(cell.x, cell.y, PheromoneMap.PREY_SCENT, 0.2)
            # Danger (predators leave danger)
            if cell.body_genome.diet > 0.5:
                pm.deposit(cell.x, cell.y, PheromoneMap.DANGER, 0.3)
            # Mate pheromone: any cell wanting to mate emits some
            if cell.mate_intent:
                pm.deposit(cell.x, cell.y, PheromoneMap.MATE, 0.3)
            # Vocalize + mate = strong pheromone cloud (calling for mates)
            if cell.vocalize_intent and cell.mate_intent:
                pm.deposit_cloud(cell.x, cell.y, PheromoneMap.MATE, 1.0, 3)

    def _regrow_food(self):
        """Regrow food at oases."""
        if len(self.foods) >= MAX_FOOD:
            return
        for ox, oy, oid in self.oases:
            # Count existing food near this oasis
            count = sum(1 for f in self.foods if f.oasis_id == oid)
            if count < FOOD_PER_OASIS:
                fx = ox + random.gauss(0, OASIS_RADIUS * 0.4)
                fy = oy + random.gauss(0, OASIS_RADIUS * 0.4)
                fx = max(10, min(self.width - 10, fx))
                fy = max(10, min(self.height - 10, fy))
                self.foods.append(Food(fx, fy, FOOD_ENERGY, oid))
                # Deposit food pheromone
                self.pheromone_map.deposit(fx, fy, PheromoneMap.FOOD_HERE, 1.0)

    def _age_carcasses(self):
        """Age carcasses and emit scent."""
        for c in self.carcasses:
            c.age += 1
            # Emit corpse scent
            if c.age % 5 == 0:
                self.pheromone_map.deposit_cloud(c.x, c.y, PheromoneMap.CORPSE_SCENT, 0.3, 2)
            # Decay
            if c.age > 300:
                c.energy -= 0.5
        self.carcasses = [c for c in self.carcasses if c.energy > 0]

    def _check_disease(self):
        """Check for disease outbreaks in dominant species."""
        living = [c for c in self.cells if c.alive]
        if len(living) < 20:
            return
        # Count by species
        species_counts = defaultdict(int)
        for c in living:
            species_counts[c.species_id] += 1
        total = len(living)
        for sid, count in species_counts.items():
            if count / total > DISEASE_OUTBREAK_THRESHOLD:
                if random.random() < DISEASE_OUTBREAK_CHANCE:
                    # Outbreak in this species
                    members = [c for c in living if c.species_id == sid and not c.sick and c.immunity_timer <= 0]
                    if members:
                        patient_zero = random.choice(members)
                        patient_zero.sick = True
                        patient_zero.sick_timer = DISEASE_DURATION

    def _spread_disease(self):
        """Spread disease between nearby cells of same species."""
        sick_cells = [c for c in self.cells if c.alive and c.sick]
        for cell in sick_cells:
            nearby = self.spatial.query(cell.x, cell.y, DISEASE_SPREAD_RANGE)
            for other in nearby:
                if not isinstance(other, Cell) or other is cell:
                    continue
                if not other.alive or other.sick or other.immunity_timer > 0:
                    continue
                if other.species_id != cell.species_id:
                    continue
                if random.random() < DISEASE_SPREAD_CHANCE:
                    other.sick = True
                    other.sick_timer = DISEASE_DURATION

    def _process_deaths(self):
        """Handle dead cells → carcasses."""
        for cell in self.cells:
            if cell.alive and cell.hp <= 0:
                cell.die()
            if not cell.alive and cell.id in self.lineage:
                if self.lineage[cell.id]['death_tick'] is None:
                    self.lineage[cell.id]['death_tick'] = self.tick
                    self.total_died += 1
                    # Create carcass
                    energy = cell.energy * CORPSE_ENERGY_RATIO + cell.body_genome.size * 2
                    if energy > 1:
                        self.carcasses.append(Carcass(cell.x, cell.y, energy, cell.body_genome.size))
                        self.pheromone_map.deposit_cloud(cell.x, cell.y, PheromoneMap.CORPSE_SCENT, 1.0, 3)

    def _extinction_respawn(self):
        """Emergency respawn when all cells die."""
        self._spawn_random_cells(EXTINCTION_RESPAWN)

    def _spawn_random_cells(self, count):
        for _ in range(count):
            x = random.randint(100, self.width - 100)
            y = random.randint(100, self.height - 100)
            cell = Cell(x, y)
            self.cells.append(cell)
            self._record_lineage(cell)

    def _record_population(self):
        living = [c for c in self.cells if c.alive]
        herbs = sum(1 for c in living if c.body_genome.diet < 0.3)
        preds = sum(1 for c in living if c.body_genome.diet > 0.7)
        omnis = len(living) - herbs - preds
        self.pop_history.append((self.tick, herbs, preds, omnis))
        if len(self.pop_history) > 2000:
            self.pop_history.pop(0)

    def living_cells(self):
        return [c for c in self.cells if c.alive]

    def living_count(self):
        return sum(1 for c in self.cells if c.alive)


# ═══════════════════════════════════════════════════════════════
#  RENDERER
# ═══════════════════════════════════════════════════════════════

class Renderer:
    """Draws the simulation world, HUD, and all screens."""

    def __init__(self, screen, world_w, world_h):
        self.screen = screen
        self.screen_w = screen.get_width()
        self.screen_h = screen.get_height()
        self.world_w = world_w
        self.world_h = world_h

        # Camera
        self.cam_x = world_w / 2 - self.screen_w / 2
        self.cam_y = world_h / 2 - self.screen_h / 2
        self.zoom = 1.0

        # Selection
        self.selected_cell = None

        # Action cam
        self.action_cam = False
        self.action_target = None

        # Fonts
        self.font_small = pygame.font.SysFont("consolas", 13)
        self.font_med = pygame.font.SysFont("consolas", 16)
        self.font_large = pygame.font.SysFont("consolas", 22)
        self.font_title = pygame.font.SysFont("consolas", 28, bold=True)

        # Colors
        self.bg_color = (15, 18, 25)
        self.grid_color = (25, 30, 40)
        self.water_color = (30, 80, 180, 120)
        self.shelter_color = (60, 50, 40)
        self.obstacle_color = (80, 75, 70)

        # Cached species lookup
        self._species_cache = {}
        self._species_cache_tick = -1

    def world_to_screen(self, wx, wy):
        return (int((wx - self.cam_x) * self.zoom),
                int((wy - self.cam_y) * self.zoom))

    def screen_to_world(self, sx, sy):
        return (sx / self.zoom + self.cam_x,
                sy / self.zoom + self.cam_y)

    def move_camera(self, dx, dy):
        self.cam_x += dx / self.zoom
        self.cam_y += dy / self.zoom
        self.cam_x = max(0, min(self.world_w - self.screen_w / self.zoom, self.cam_x))
        self.cam_y = max(0, min(self.world_h - self.screen_h / self.zoom, self.cam_y))

    def zoom_camera(self, factor, mx=None, my=None):
        old_zoom = self.zoom
        self.zoom = max(0.3, min(3.0, self.zoom * factor))
        if mx is not None:
            # Zoom toward mouse
            wx, wy = self.screen_to_world(mx, my)
            self.cam_x = wx - mx / self.zoom
            self.cam_y = wy - my / self.zoom

    def handle_click(self, pos, world):
        wx, wy = self.screen_to_world(*pos)
        # Find nearest cell
        best = None
        best_d = 30  # click threshold
        for cell in world.cells:
            if not cell.alive:
                continue
            d = math.hypot(cell.x - wx, cell.y - wy)
            if d < best_d:
                best_d = d
                best = cell
        self.selected_cell = best

    def update_action_cam(self, world):
        if self.action_cam and self.action_target:
            if self.action_target.alive:
                self.cam_x = self.action_target.x - self.screen_w / (2 * self.zoom)
                self.cam_y = self.action_target.y - self.screen_h / (2 * self.zoom)

    # ──────── MAIN DRAW ────────

    def draw(self, world, paused, sim_speed):
        self.screen.fill(self.bg_color)

        self.update_action_cam(world)

        # Visible area culling bounds
        vx1 = self.cam_x - 50
        vy1 = self.cam_y - 50
        vx2 = self.cam_x + self.screen_w / self.zoom + 50
        vy2 = self.cam_y + self.screen_h / self.zoom + 50

        # Grid lines
        self._draw_grid()

        # World boundary
        bx, by = self.world_to_screen(0, 0)
        bw = int(self.world_w * self.zoom)
        bh = int(self.world_h * self.zoom)
        pygame.draw.rect(self.screen, (40, 40, 50), (bx, by, bw, bh), 2)

        # Water sources
        for wx, wy in world.water_sources:
            if vx1 < wx < vx2 and vy1 < wy < vy2:
                sx, sy = self.world_to_screen(wx, wy)
                r = int(WATER_RADIUS * self.zoom)
                water_surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                pygame.draw.circle(water_surf, (30, 80, 180, 60), (r, r), r)
                pygame.draw.circle(water_surf, (40, 100, 200, 30), (r, r), r, 2)
                self.screen.blit(water_surf, (sx - r, sy - r))

        # Shelters
        for sx, sy in world.shelters:
            if vx1 < sx < vx2 and vy1 < sy < vy2:
                px, py = self.world_to_screen(sx, sy)
                r = int(SHELTER_RADIUS * self.zoom)
                pygame.draw.circle(self.screen, self.shelter_color, (px, py), r, 1)

        # Obstacles
        for ox, oy, ow, oh in world.obstacles:
            if ox + ow > vx1 and ox < vx2 and oy + oh > vy1 and oy < vy2:
                px, py = self.world_to_screen(ox, oy)
                pygame.draw.rect(self.screen, self.obstacle_color,
                                 (px, py, int(ow * self.zoom), int(oh * self.zoom)))

        # Pheromones (low-alpha overlay)
        self._draw_pheromones(world, vx1, vy1, vx2, vy2)

        # Food
        for food in world.foods:
            if vx1 < food.x < vx2 and vy1 < food.y < vy2:
                sx, sy = self.world_to_screen(food.x, food.y)
                r = max(2, int(FOOD_RADIUS * self.zoom))
                brightness = min(255, int(100 + food.energy * 5))
                pygame.draw.circle(self.screen, (30, brightness, 30), (sx, sy), r)

        # Carcasses
        for carcass in world.carcasses:
            if vx1 < carcass.x < vx2 and vy1 < carcass.y < vy2:
                sx, sy = self.world_to_screen(carcass.x, carcass.y)
                r = max(3, int(carcass.size * 0.5 * self.zoom))
                # Reddish-brown blob with energy-based opacity
                alpha = min(255, int(80 + carcass.energy * 3))
                pygame.draw.circle(self.screen, (140, 50, 30), (sx, sy), r)
                pygame.draw.circle(self.screen, (90, 35, 20), (sx, sy), r, 1)
                # Small X mark
                xr = max(2, r // 2)
                pygame.draw.line(self.screen, (180, 70, 50), (sx - xr, sy - xr), (sx + xr, sy + xr), 1)
                pygame.draw.line(self.screen, (180, 70, 50), (sx - xr, sy + xr), (sx + xr, sy - xr), 1)

        # Cells
        for cell in world.cells:
            if not cell.alive:
                continue
            if vx1 < cell.x < vx2 and vy1 < cell.y < vy2:
                self._draw_cell(cell)

        # Selection highlight + sense overlays
        if self.selected_cell and self.selected_cell.alive:
            sc = self.selected_cell
            sx, sy = self.world_to_screen(sc.x, sc.y)
            r = int((sc.body_genome.size + 5) * self.zoom)
            pygame.draw.circle(self.screen, (255, 255, 100), (sx, sy), r, 2)

            genome = sc.body_genome
            # Vision cone (blue, ~120° forward)
            vis_r = int(genome.vision_range * self.zoom)
            if vis_r > 5:
                half_angle = math.pi / 3  # ~60° half = 120° total
                cone_pts = [(sx, sy)]
                for step in range(17):
                    a = sc.angle - half_angle + (2 * half_angle) * step / 16
                    cone_pts.append((
                        sx + int(math.cos(a) * vis_r),
                        sy + int(math.sin(a) * vis_r)))
                if len(cone_pts) >= 3:
                    cone_surf = pygame.Surface((self.screen_w, self.screen_h), pygame.SRCALPHA)
                    pygame.draw.polygon(cone_surf, (40, 100, 220, 25), cone_pts)
                    pygame.draw.polygon(cone_surf, (60, 130, 255, 80), cone_pts, 1)
                    self.screen.blit(cone_surf, (0, 0))

            # Hearing radius (orange dashed circle, 360°)
            hear_r = int(genome.hearing_range * self.zoom)
            if hear_r > 5:
                for a_deg in range(0, 360, 6):
                    a1 = math.radians(a_deg)
                    a2 = math.radians(a_deg + 3)
                    p1 = (sx + int(math.cos(a1) * hear_r), sy + int(math.sin(a1) * hear_r))
                    p2 = (sx + int(math.cos(a2) * hear_r), sy + int(math.sin(a2) * hear_r))
                    pygame.draw.line(self.screen, (220, 140, 40), p1, p2, 1)

        # HUD
        self._draw_hud(world, paused, sim_speed)

        # Cell info panel
        if self.selected_cell and self.selected_cell.alive:
            self._draw_cell_info(self.selected_cell, world)

        # Minimap
        self._draw_minimap(world)

    def _draw_grid(self):
        gs = int(100 * self.zoom)
        if gs < 20:
            return
        start_x = int(-self.cam_x * self.zoom) % gs
        start_y = int(-self.cam_y * self.zoom) % gs
        for x in range(start_x, self.screen_w, gs):
            pygame.draw.line(self.screen, self.grid_color, (x, 0), (x, self.screen_h))
        for y in range(start_y, self.screen_h, gs):
            pygame.draw.line(self.screen, self.grid_color, (0, y), (self.screen_w, y))

    def _draw_pheromones(self, world, vx1, vy1, vx2, vy2):
        pm = world.pheromone_map
        gs = pm.grid_size
        z = self.zoom
        for r in range(pm.rows):
            wy = r * gs
            if wy < vy1 - gs or wy > vy2:
                continue
            for c in range(pm.cols):
                wx = c * gs
                if wx < vx1 - gs or wx > vx2:
                    continue
                vals = pm.data[r, c]
                total = vals.sum()
                if total < 0.1:
                    continue
                sx, sy = self.world_to_screen(wx, wy)
                sz = max(2, int(gs * z))
                # Color blend
                red = min(255, int(vals[PheromoneMap.DANGER] * 50 + vals[PheromoneMap.CORPSE_SCENT] * 30))
                green = min(255, int(vals[PheromoneMap.FOOD_HERE] * 50 + vals[PheromoneMap.TRAIL] * 15))
                blue = min(255, int(vals[PheromoneMap.MATE] * 40 + vals[PheromoneMap.PREY_SCENT] * 25))
                alpha = min(80, int(total * 15))
                if alpha > 5:
                    surf = pygame.Surface((sz, sz), pygame.SRCALPHA)
                    surf.fill((red, green, blue, alpha))
                    self.screen.blit(surf, (sx, sy))

    def _draw_cell(self, cell):
        bg = cell.body_genome
        sx, sy = self.world_to_screen(cell.x, cell.y)
        r = max(3, int(bg.size * self.zoom))
        color = bg.color()
        scr = self.screen
        z = self.zoom

        # --- Trail ---
        if len(cell.trail) > 1 and z > 0.5:
            n = len(cell.trail)
            for i in range(1, n):
                p1 = self.world_to_screen(cell.trail[i - 1][0], cell.trail[i - 1][1])
                p2 = self.world_to_screen(cell.trail[i][0], cell.trail[i][1])
                pygame.draw.line(scr, color, p1, p2, 1)

        # --- Cilia (real thrust direction, no random wave) ---
        cilia_angles = bg.cilia_positions()
        n_cilia = bg.num_cilia
        cilia_base_color = tuple(min(255, c + 80) for c in color)
        for i in range(n_cilia):
            ca = cilia_angles[i]
            power = cell.cilia_powers[i]         # 0..1
            steer = cell.cilia_steers[i]

            # Attachment point on cell surface
            attach_angle = cell.angle + ca
            # Cilium visual direction = push direction (opposite of cilia pointing)
            push_angle = attach_angle + steer + math.pi
            # Paddle stroke animation: oscillate along push axis based on power
            stroke = math.sin(cell.cilia_phase + i * 1.8) * power * 0.2
            draw_angle = push_angle + stroke

            # Length proportional to power
            base_len = r * 0.5 + 2
            cilia_len = max(2, int((base_len + power * base_len * 0.8)))

            # Width based on power
            cilia_w = max(1, int(z * (1 + power * 0.5)))

            # Color: brighter green when pushing harder, dim when idle
            green_boost = int(power * 80)
            cilia_color = (max(0, cilia_base_color[0] - green_boost // 2),
                           min(255, cilia_base_color[1] + green_boost),
                           max(0, cilia_base_color[2] - green_boost // 2))

            # Draw from cell surface outward in push direction
            cx1 = sx + int(math.cos(attach_angle) * r)
            cy1 = sy + int(math.sin(attach_angle) * r)
            cx2 = cx1 + int(math.cos(draw_angle) * cilia_len * z)
            cy2 = cy1 + int(math.sin(draw_angle) * cilia_len * z)
            pygame.draw.line(scr, cilia_color, (cx1, cy1), (cx2, cy2), cilia_w)

        # --- Horns (multi-horn, gene-positioned) ---
        horn_sz = bg.horn_size
        n_horns = bg.num_horns
        if horn_sz > 0.3 and n_horns > 0 and r > 3:
            horn_angles = bg.horn_positions()
            horn_len = max(2, int((horn_sz * 2.5 + 2) * z))
            horn_w = max(1, int(z * (1 + horn_sz * 0.3)))
            horn_brightness = min(255, 200 + int(horn_sz * 10))
            horn_color = (horn_brightness, horn_brightness - 20, max(0, horn_brightness - 60))
            for ha in horn_angles:
                h_angle = cell.angle + ha
                hx1 = sx + int(math.cos(h_angle) * r * 0.8)
                hy1 = sy + int(math.sin(h_angle) * r * 0.8)
                hx2 = hx1 + int(math.cos(h_angle) * horn_len)
                hy2 = hy1 + int(math.sin(h_angle) * horn_len)
                pygame.draw.line(scr, horn_color, (hx1, hy1), (hx2, hy2), horn_w)
                if horn_sz > 2.0:
                    pygame.draw.circle(scr, (255, 240, 200), (hx2, hy2), max(1, horn_w))

        # --- Cell body ---
        pygame.draw.circle(scr, color, (sx, sy), r)

        # Inner core (nucleus) — brighter
        inner_r = max(1, r // 2)
        brighter = tuple(min(255, c + 50) for c in color)
        pygame.draw.circle(scr, brighter, (sx, sy), inner_r)

        # --- Mouth (diet-dependent visual at front) ---
        diet = bg.diet
        if r > 3:
            mouth_sz = bg.mouth_size
            mouth_w = max(2, int(mouth_sz * r * 0.3 * z))
            mouth_arc = bg.mouth_cone_half  # radians

            if diet > 0.7:
                # Carnivore: V-shaped jaw with teeth (red)
                mouth_color = (220, 60, 50)
                # Two jaw lines forming a V
                for side in (-1, 1):
                    jaw_angle = cell.angle + side * mouth_arc * 0.6
                    jx1 = sx + int(math.cos(jaw_angle) * r * 0.7)
                    jy1 = sy + int(math.sin(jaw_angle) * r * 0.7)
                    jx2 = sx + int(math.cos(cell.angle) * (r + mouth_w))
                    jy2 = sy + int(math.sin(cell.angle) * (r + mouth_w))
                    pygame.draw.line(scr, mouth_color, (jx1, jy1), (jx2, jy2), max(1, int(z * 1.2)))
                # Teeth: small triangles along jaw edge
                n_teeth = max(2, min(5, int(mouth_sz * 2)))
                for t in range(n_teeth):
                    ta = cell.angle + (t - n_teeth / 2 + 0.5) * mouth_arc * 0.3
                    tx = sx + int(math.cos(ta) * (r + 1))
                    ty = sy + int(math.sin(ta) * (r + 1))
                    tx2 = sx + int(math.cos(ta) * (r + max(1, int(mouth_sz * 1.5 * z))))
                    ty2 = sy + int(math.sin(ta) * (r + max(1, int(mouth_sz * 1.5 * z))))
                    pygame.draw.line(scr, (255, 240, 230), (tx, ty), (tx2, ty2), 1)
            elif diet < 0.3:
                # Herbivore: wide arc mouth (green)
                mouth_color = (60, 200, 80)
                # Draw arc as segmented line at front
                n_seg = 6
                pts = []
                for s in range(n_seg + 1):
                    frac = (s / n_seg - 0.5) * 2  # -1..1
                    a = cell.angle + frac * mouth_arc * 0.7
                    px = sx + int(math.cos(a) * (r + max(1, int(mouth_sz * 0.8 * z))))
                    py = sy + int(math.sin(a) * (r + max(1, int(mouth_sz * 0.8 * z))))
                    pts.append((px, py))
                if len(pts) > 1:
                    pygame.draw.lines(scr, mouth_color, False, pts, max(1, int(z * 1.5)))
                # Lip dots at edges
                for end_pt in (pts[0], pts[-1]):
                    pygame.draw.circle(scr, mouth_color, end_pt, max(1, int(z)))
            else:
                # Omnivore: medium arc with 1-2 small teeth (orange)
                mouth_color = (230, 160, 40)
                n_seg = 4
                pts = []
                for s in range(n_seg + 1):
                    frac = (s / n_seg - 0.5) * 2
                    a = cell.angle + frac * mouth_arc * 0.5
                    px = sx + int(math.cos(a) * (r + max(1, int(mouth_sz * 0.6 * z))))
                    py = sy + int(math.sin(a) * (r + max(1, int(mouth_sz * 0.6 * z))))
                    pts.append((px, py))
                if len(pts) > 1:
                    pygame.draw.lines(scr, mouth_color, False, pts, max(1, int(z)))
                # 1-2 small teeth
                for t in range(min(2, max(1, int(mouth_sz)))):
                    ta = cell.angle + (t - 0.5) * mouth_arc * 0.2
                    tx = sx + int(math.cos(ta) * (r + 1))
                    ty = sy + int(math.sin(ta) * (r + 1))
                    tx2 = sx + int(math.cos(ta) * (r + max(1, int(mouth_sz * z))))
                    ty2 = sy + int(math.sin(ta) * (r + max(1, int(mouth_sz * z))))
                    pygame.draw.line(scr, (240, 230, 210), (tx, ty), (tx2, ty2), 1)

        # Direction indicator (eye dot)
        if r > 4:
            dir_x = sx + int(math.cos(cell.angle) * r * 0.6)
            dir_y = sy + int(math.sin(cell.angle) * r * 0.6)
            pygame.draw.circle(scr, (255, 255, 255), (dir_x, dir_y), max(1, r // 5))

        # Diet border: carnivore=red, omnivore=orange, herbivore=green
        if diet > 0.7:
            pygame.draw.circle(scr, (255, 60, 60), (sx, sy), r, 1)
        elif diet > 0.3:
            pygame.draw.circle(scr, (255, 180, 40), (sx, sy), r, 1)
        else:
            pygame.draw.circle(scr, (80, 200, 80), (sx, sy), r, 1)

        # --- Status overlays ---

        # Disease: pulsing green-yellow aura
        if cell.sick:
            pulse = 0.5 + 0.5 * math.sin(cell.age * 0.15)
            sick_r = r + max(2, int(4 * z * pulse))
            sick_alpha = int(40 + 40 * pulse)
            sick_surf = pygame.Surface((sick_r * 2 + 4, sick_r * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(sick_surf, (80, 200, 30, sick_alpha),
                               (sick_r + 2, sick_r + 2), sick_r)
            scr.blit(sick_surf, (sx - sick_r - 2, sy - sick_r - 2))

        # Juvenile: light blue contour
        if cell.juvenile:
            pygame.draw.circle(scr, (180, 180, 255), (sx, sy), r + 2, 1)

        # High exertion: flashing bright ring when cilia are pushing hard
        n_c = bg.num_cilia
        avg_power = sum(cell.cilia_powers[i] for i in range(n_c)) / max(n_c, 1)
        if avg_power > 0.7:
            sprint_color = (255, 255, 100) if diet > 0.5 else (100, 255, 255)
            pygame.draw.circle(scr, sprint_color, (sx, sy), r + 3, 1)

        # Attack intent: red ring
        if cell.attack_intent:
            pygame.draw.circle(scr, (255, 50, 50), (sx, sy), r + 2, 1)

        # Mate intent: pulsing pink aura
        if cell.mate_intent and not cell.juvenile and r > 3:
            pulse = abs(math.sin(cell.age * 0.08)) * 0.5 + 0.5
            mate_r = int(r + 3 + pulse * 3)
            mate_surf = pygame.Surface((mate_r * 2, mate_r * 2), pygame.SRCALPHA)
            pygame.draw.circle(mate_surf, (255, 100, 180, int(40 + pulse * 40)),
                               (mate_r, mate_r), mate_r, max(1, int(2 * z)))
            scr.blit(mate_surf, (sx - mate_r, sy - mate_r))

        # Damage flash: red flicker
        if cell.damaged_flash > 0:
            flash_alpha = min(180, cell.damaged_flash * 30)
            flash_surf = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(flash_surf, (255, 50, 30, flash_alpha), (r + 2, r + 2), r + 2)
            scr.blit(flash_surf, (sx - r - 2, sy - r - 2))

        # Vocalize: expanding sonar ring
        if cell.vocalize_intent:
            voc_r = int((r + 10 + (cell.age % 20)) * z)
            voc_alpha = max(0, 80 - (cell.age % 20) * 4)
            if voc_alpha > 5 and voc_r > 2:
                voc_surf = pygame.Surface((voc_r * 2, voc_r * 2), pygame.SRCALPHA)
                voc_color = (255, 100, 180, voc_alpha) if cell.mate_intent else (100, 200, 255, voc_alpha)
                pygame.draw.circle(voc_surf, voc_color, (voc_r, voc_r), voc_r, max(1, int(z)))
                scr.blit(voc_surf, (sx - voc_r, sy - voc_r))

        # --- HP & Energy bars ---
        if r > 4:
            bar_w = r * 2
            bar_h = 2
            hp_ratio = cell.hp / bg.max_hp if bg.max_hp > 0 else 1.0
            if hp_ratio < 1.0:
                pygame.draw.rect(scr, (40, 40, 40),
                                 (sx - bar_w // 2, sy - r - 9, bar_w, bar_h))
                hp_color = (200, 50, 50) if hp_ratio < 0.3 else (220, 160, 30) if hp_ratio < 0.6 else (50, 200, 50)
                pygame.draw.rect(scr, hp_color,
                                 (sx - bar_w // 2, sy - r - 9, int(bar_w * hp_ratio), bar_h))

            energy_ratio = min(1, cell.energy / max(1, bg.repro_threshold))
            pygame.draw.rect(scr, (30, 30, 30),
                             (sx - bar_w // 2, sy - r - 6, bar_w, bar_h))
            e_color = (50, 180, 50) if energy_ratio > 0.3 else (180, 50, 50)
            pygame.draw.rect(scr, e_color,
                             (sx - bar_w // 2, sy - r - 6, int(bar_w * energy_ratio), bar_h))

    def _draw_hud(self, world, paused, sim_speed):
        living = world.living_count()
        y = 8
        texts = [
            f"Tick: {world.tick}  Pop: {living}  Speed: {sim_speed}x",
            f"Food: {len(world.foods)}  Carcasses: {len(world.carcasses)}",
            f"Species: {len(world.neat_pop.species)}  Born: {world.total_born}  Died: {world.total_died}",
        ]
        if paused:
            texts[0] += "  [PAUSED]"

        for t in texts:
            surf = self.font_small.render(t, True, (180, 190, 200))
            self.screen.blit(surf, (8, y))
            y += 16

        # Population graph (small, top-right)
        self._draw_pop_graph(world)

    def _draw_pop_graph(self, world):
        if len(world.pop_history) < 2:
            return
        gw, gh = 200, 60
        gx = self.screen_w - gw - 10
        gy = 8

        # Background
        bg_surf = pygame.Surface((gw, gh), pygame.SRCALPHA)
        bg_surf.fill((0, 0, 0, 120))
        self.screen.blit(bg_surf, (gx, gy))

        history = world.pop_history[-200:]
        max_pop = max(max(h + p + o for _, h, p, o in history), 1)

        n = len(history)
        step = gw / max(n - 1, 1)

        # Draw lines for each type
        for type_idx, color in [(1, (50, 200, 50)), (2, (200, 50, 50)), (3, (200, 200, 50))]:
            points = []
            for i, entry in enumerate(history):
                px = gx + int(i * step)
                val = entry[type_idx]
                py = gy + gh - int(val / max_pop * gh)
                points.append((px, py))
            if len(points) > 1:
                pygame.draw.lines(self.screen, color, False, points, 1)

    def _draw_cell_info(self, cell, world):
        """Draw info panel for selected cell."""
        bg = cell.body_genome
        pw, ph = 260, 500
        px = self.screen_w - pw - 10
        py = 80

        # Panel background
        panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
        panel.fill((10, 12, 18, 200))
        pygame.draw.rect(panel, (50, 55, 65), (0, 0, pw, ph), 1)
        self.screen.blit(panel, (px, py))

        x = px + 8
        y = py + 8
        lh = 16

        # Species name
        sp_name = self._get_species_name(cell, world)
        title = self.font_med.render(f"✧ {sp_name}", True, (220, 200, 160))
        self.screen.blit(title, (x, y))
        y += lh + 4

        # Diet label & color swatch
        diet_label = bg.diet_label()
        color = bg.color()
        pygame.draw.circle(self.screen, color, (x + 8, y + 7), 6)
        info = self.font_small.render(f"  {diet_label.capitalize()} (diet={bg.diet:.2f})", True, (180, 180, 180))
        self.screen.blit(info, (x + 16, y))
        y += lh

        def _line(label, value, col=(160, 170, 180)):
            nonlocal y
            s = self.font_small.render(f"{label}: {value}", True, col)
            self.screen.blit(s, (x, y))
            y += lh

        _line("ID", f"{cell.id} (gen {cell.generation})")
        _line("Age", f"{cell.age} ticks")
        _line("Energy", f"{cell.energy:.1f} / {bg.repro_threshold:.0f} (cap {bg.energy_cap:.0f})")
        _line("HP", f"{cell.hp:.1f} / {bg.max_hp:.1f}")
        _line("Thirst", f"{cell.thirst:.1f} / {THIRST_MAX:.0f}")
        _line("Speed", f"{cell.speed:.2f} / {bg.max_speed:.2f}")
        y += 4

        _line("Size", f"{bg.size:.1f}")
        _line("Vision", f"{bg.vision_range:.0f}")
        _line("Hearing", f"{bg.hearing_range:.0f}")
        _line("Horn", f"{bg.num_horns}x sz{bg.horn_size:.1f} (spr={bg.horn_spread:.2f})")
        _line("Mouth", f"{bg.mouth_size:.1f} (cone={math.degrees(bg.mouth_cone_half):.0f}°)")
        _line("Turn rate", f"{bg.turn_rate:.3f} rad/tick")
        _line("Metabolism", f"{bg.metabolism:.2f}")
        _line("Attack", f"{bg.attack_power:.1f} / Def={bg.horn_defense:.1f}")
        y += 4

        # Brain stats
        n_hidden = cell.neat_genome.num_hidden()
        n_conns = cell.neat_genome.num_enabled_connections()
        brain_cost = cell.neat_genome.brain_cost()
        _line("Brain", f"{n_hidden}h / {n_conns}c ({brain_cost:.3f}/tick)", (140, 180, 220))
        _line("Fitness", f"{cell.fitness:.1f}")
        _line("Species", f"#{cell.species_id}")
        y += 4

        _line("Children", str(cell.children_count))
        _line("Kills", str(cell.kills))
        _line("Food eaten", str(cell.food_eaten))

        # State flags
        flags = []
        if cell.juvenile:
            flags.append("JUV")
        if cell.sick:
            flags.append("SICK")
        nc = cell.body_genome.num_cilia
        if sum(cell.cilia_powers[i] for i in range(nc)) / max(nc, 1) > 0.7:
            flags.append("EXERT")
        if cell.attack_intent:
            flags.append("ATK")
        if cell.eat_intent:
            flags.append("EAT")
        if cell.mate_intent:
            flags.append("MATE")
        if flags:
            y += 2
            _line("State", " ".join(flags), (200, 200, 100))

        # Energy drain breakdown
        y += 4
        body_drain = bg.base_energy_drain()
        _line("Drain/tick", f"body={body_drain:.4f} brain={brain_cost:.4f}", (120, 130, 140))

    def _get_species_name(self, cell, world):
        """Get or generate species name for a cell."""
        if cell.species_id < 0:
            return "Unclassified"
        if self._species_cache_tick != world.tick:
            self._species_cache = {}
            self._species_cache_tick = world.tick
        if cell.species_id not in self._species_cache:
            self._species_cache[cell.species_id] = world.neat_pop.generate_latin_name(cell.species_id)
        return self._species_cache[cell.species_id]

    def _draw_minimap(self, world):
        mw, mh = 150, 100
        mx = self.screen_w - mw - 10
        my = self.screen_h - mh - 10

        # Background
        mm_surf = pygame.Surface((mw, mh), pygame.SRCALPHA)
        mm_surf.fill((0, 0, 0, 150))

        sx = mw / world.width
        sy = mh / world.height

        # Food dots
        for food in world.foods:
            px = int(food.x * sx)
            py = int(food.y * sy)
            mm_surf.set_at((max(0, min(mw - 1, px)), max(0, min(mh - 1, py))), (50, 200, 50, 200))

        # Cell dots
        for cell in world.cells:
            if not cell.alive:
                continue
            px = int(cell.x * sx)
            py = int(cell.y * sy)
            color = cell.body_genome.color()
            pygame.draw.circle(mm_surf, (*color, 220), (max(1, min(mw - 2, px)), max(1, min(mh - 2, py))), 2)

        # Viewport rectangle
        vx1 = int(self.cam_x * sx)
        vy1 = int(self.cam_y * sy)
        vw = int(self.screen_w / self.zoom * sx)
        vh = int(self.screen_h / self.zoom * sy)
        pygame.draw.rect(mm_surf, (200, 200, 200, 150), (vx1, vy1, vw, vh), 1)

        self.screen.blit(mm_surf, (mx, my))
        pygame.draw.rect(self.screen, (50, 55, 65), (mx, my, mw, mh), 1)

    def minimap_click(self, mouse_pos, world):
        mw, mh = 150, 100
        mx = self.screen_w - mw - 10
        my = self.screen_h - mh - 10
        if mx <= mouse_pos[0] <= mx + mw and my <= mouse_pos[1] <= my + mh:
            rx = (mouse_pos[0] - mx) / mw
            ry = (mouse_pos[1] - my) / mh
            self.cam_x = rx * world.width - self.screen_w / (2 * self.zoom)
            self.cam_y = ry * world.height - self.screen_h / (2 * self.zoom)
            return True
        return False

    # ──────── SPECIES SCREEN (F6) ────────

    def draw_species_screen(self, world):
        self.screen.fill((10, 12, 18))
        y = 10
        title = self.font_title.render("Species (F6)", True, (200, 210, 220))
        self.screen.blit(title, (20, y))
        y += 40

        species_list = sorted(world.neat_pop.species.values(), key=lambda s: -len(s.members))

        for sp in species_list:
            if y > self.screen_h - 30:
                break
            name = world.neat_pop.generate_latin_name(sp.id)
            count = len(sp.members)
            avg_fit = sp.avg_fitness
            stag = sp.stagnation_counter

            # Color from first member
            color = (150, 150, 150)
            if sp.members:
                color = sp.members[0].body_genome.color()

            pygame.draw.circle(self.screen, color, (30, y + 8), 6)
            text = f"  {name} (#{sp.id})  Pop: {count}  Fit: {avg_fit:.1f}  Stag: {stag}"
            surf = self.font_small.render(text, True, (180, 190, 200))
            self.screen.blit(surf, (42, y))

            # Highlight if selected cell is in this species
            if self.selected_cell and self.selected_cell.alive and self.selected_cell.species_id == sp.id:
                pygame.draw.rect(self.screen, (255, 255, 100), (18, y - 2, self.screen_w - 36, 20), 1)

            y += 22

        # Instructions
        instr = self.font_small.render("Press F6 to return | Click species to select member", True, (100, 110, 120))
        self.screen.blit(instr, (20, self.screen_h - 25))

    # ──────── BRAIN SCREEN (F7) ────────

    def draw_brain_screen(self, world):
        self.screen.fill((10, 12, 18))
        y = 10
        title = self.font_title.render("Brain (F7) — NEAT Topology", True, (200, 210, 220))
        self.screen.blit(title, (20, y))
        y += 40

        cell = self.selected_cell
        if not cell or not cell.alive:
            msg = self.font_med.render("Select a cell to view its brain", True, (120, 130, 140))
            self.screen.blit(msg, (20, y))
            return

        # Cell info
        name = self._get_species_name(cell, world)
        info = self.font_med.render(
            f"Cell #{cell.id} — {name} — {cell.neat_genome.num_hidden()}h "
            f"{cell.neat_genome.num_enabled_connections()}c — "
            f"cost: {cell.neat_genome.brain_cost():.3f}/tick",
            True, (180, 190, 200)
        )
        self.screen.blit(info, (20, y))
        y += 30

        # Draw network topology
        self._draw_neat_topology(cell, 40, y, self.screen_w - 80, self.screen_h - y - 120)

        # Body gene bars at bottom
        self._draw_body_gene_bars(cell, 40, self.screen_h - 100, self.screen_w - 80)

        # Instructions
        instr = self.font_small.render("Press F7 to return", True, (100, 110, 120))
        self.screen.blit(instr, (20, self.screen_h - 25))

    def _draw_neat_topology(self, cell, x, y, w, h):
        """Draw the actual NEAT network topology with Kahn layers.
        Features: Bézier curves, alpha-blended weight lines, hover highlight."""
        genome = cell.neat_genome
        net = cell.get_compiled_network()
        if not net or not net.valid:
            return

        layers = net.layers  # node_id → layer number
        if not layers:
            return

        max_layer = max(layers.values()) if layers else 0
        if max_layer == 0:
            max_layer = 1

        # --- Layout with proper margins for labels ---
        left_margin = 100   # space for input labels
        right_margin = 90   # space for output labels
        draw_x = x + left_margin
        draw_w = w - left_margin - right_margin

        # Group nodes by layer
        layer_nodes = defaultdict(list)
        for nid, layer in layers.items():
            layer_nodes[layer].append(nid)

        # Sort nodes within layers
        for layer in layer_nodes:
            layer_nodes[layer].sort()

        # Compute positions
        node_pos = {}
        for layer, nodes in layer_nodes.items():
            lx = draw_x + int(layer / max_layer * draw_w)
            n = len(nodes)
            for i, nid in enumerate(nodes):
                ny = y + int((i + 1) / (n + 1) * h)
                node_pos[nid] = (lx, ny)

        # --- Hover detection ---
        mx, my = pygame.mouse.get_pos()
        hovered_node = None
        hover_dist = 15
        for nid, pos in node_pos.items():
            d = math.hypot(pos[0] - mx, pos[1] - my)
            if d < hover_dist:
                hover_dist = d
                hovered_node = nid

        # Build set of connections touching hovered node
        hover_conns = set()
        hover_nodes = set()
        if hovered_node is not None:
            hover_nodes.add(hovered_node)
            for innov, (in_n, out_n, weight, enabled) in genome.connections.items():
                if enabled and (in_n == hovered_node or out_n == hovered_node):
                    hover_conns.add(innov)
                    hover_nodes.add(in_n)
                    hover_nodes.add(out_n)

        # --- Draw connections as Bézier curves ---
        # Use SRCALPHA surface for proper transparency
        conn_surf = pygame.Surface((self.screen_w, self.screen_h), pygame.SRCALPHA)

        for innov, (in_n, out_n, weight, enabled) in genome.connections.items():
            if in_n not in node_pos or out_n not in node_pos:
                continue
            p1 = node_pos[in_n]
            p2 = node_pos[out_n]

            if not enabled:
                # Skip disabled connections unless hovered
                if innov in hover_conns:
                    self._draw_bezier(conn_surf, p1, p2, (60, 60, 60, 100), 1)
                continue

            aw = abs(weight)
            is_highlight = innov in hover_conns

            if hovered_node is not None and not is_highlight:
                # Dim non-related connections when hovering
                alpha = int(min(40, aw * 15 + 5))
                thickness = 1
            else:
                # Normal: alpha and thickness from weight
                alpha = int(min(200, aw * 60 + 20))
                thickness = max(1, min(4, int(aw * 1.5)))

            if is_highlight:
                alpha = 255
                thickness = max(2, min(5, int(aw * 2)))

            if weight > 0:
                color = (50, 200, 80, alpha)
            else:
                color = (220, 50, 50, alpha)

            self._draw_bezier(conn_surf, p1, p2, color, thickness)

            # Weight label on highlighted connections
            if is_highlight:
                mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                wlabel = self.font_small.render(f"{weight:.2f}", True, (220, 220, 220))
                bg_r = pygame.Rect(mid[0] - 2, mid[1] - 8, wlabel.get_width() + 4, 16)
                pygame.draw.rect(conn_surf, (20, 20, 30, 200), bg_r)
                conn_surf.blit(wlabel, (mid[0], mid[1] - 7))

        self.screen.blit(conn_surf, (0, 0))

        # --- Memory loop arrows (blue dashed) ---
        mem_pairs = [(NUM_INPUTS + OUT_MEM_0, IN_MEM_0),
                     (NUM_INPUTS + OUT_MEM_1, IN_MEM_1)]
        for src, dst in mem_pairs:
            if src in node_pos and dst in node_pos:
                p1 = node_pos[src]
                p2 = node_pos[dst]
                # Curved path looping below
                mid_y = max(p1[1], p2[1]) + 40
                mid_x = (p1[0] + p2[0]) // 2
                points = [p1, (p1[0] + 30, mid_y), (mid_x, mid_y + 15),
                          (p2[0] - 30, mid_y), p2]
                for i in range(len(points) - 1):
                    pygame.draw.line(self.screen, (60, 100, 180), points[i], points[i + 1], 2)
                # Arrow head
                dx = p2[0] - points[-2][0]
                dy = p2[1] - points[-2][1]
                d = math.sqrt(dx * dx + dy * dy) + 0.001
                ax = int(p2[0] - dx / d * 8 - dy / d * 4)
                ay = int(p2[1] - dy / d * 8 + dx / d * 4)
                bx = int(p2[0] - dx / d * 8 + dy / d * 4)
                by = int(p2[1] - dy / d * 8 - dx / d * 4)
                pygame.draw.polygon(self.screen, (60, 100, 180), [p2, (ax, ay), (bx, by)])
                # Label
                lbl = self.font_small.render("memory", True, (60, 100, 180))
                self.screen.blit(lbl, (mid_x - lbl.get_width() // 2, mid_y + 2))

        # --- Draw nodes ---
        for nid, pos in node_pos.items():
            ntype = genome.nodes.get(nid, (NODE_HIDDEN, ACT_TANH))[0]
            is_hl = nid in hover_nodes and hovered_node is not None

            if ntype == NODE_INPUT:
                color = (80, 150, 220)
                r = 7 if is_hl else 5
            elif ntype == NODE_OUTPUT:
                color = (220, 160, 60)
                r = 7 if is_hl else 5
            else:
                color = (100, 220, 120)
                r = 6 if is_hl else 4

            if is_hl:
                # Glow ring
                glow_surf = pygame.Surface((r * 4 + 4, r * 4 + 4), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (*color, 40), (r * 2 + 2, r * 2 + 2), r * 2)
                self.screen.blit(glow_surf, (pos[0] - r * 2 - 2, pos[1] - r * 2 - 2))

            pygame.draw.circle(self.screen, color, pos, r)
            if is_hl:
                pygame.draw.circle(self.screen, (255, 255, 255), pos, r, 2)
            else:
                pygame.draw.circle(self.screen, (180, 180, 180), pos, r, 1)

            # Labels
            if ntype == NODE_INPUT and nid < len(INPUT_NAMES):
                label = INPUT_NAMES[nid]
                lc = (140, 180, 220) if is_hl else (80, 110, 150)
                surf = self.font_small.render(label, True, lc)
                self.screen.blit(surf, (pos[0] - surf.get_width() - 10, pos[1] - 7))
            elif ntype == NODE_OUTPUT:
                idx = nid - NUM_INPUTS
                if 0 <= idx < len(OUTPUT_NAMES):
                    label = OUTPUT_NAMES[idx]
                    lc = (230, 180, 70) if is_hl else (150, 120, 60)
                    surf = self.font_small.render(label, True, lc)
                    self.screen.blit(surf, (pos[0] + 10, pos[1] - 7))
            elif ntype == NODE_HIDDEN:
                lc = (150, 230, 150) if is_hl else (80, 140, 80)
                surf = self.font_small.render(f"h{nid}", True, lc)
                self.screen.blit(surf, (pos[0] - surf.get_width() // 2, pos[1] - 18))

        # --- Hover tooltip ---
        if hovered_node is not None:
            ntype = genome.nodes.get(hovered_node, (NODE_HIDDEN, ACT_TANH))[0]
            n_in = sum(1 for c in genome.connections.values() if c[3] and c[1] == hovered_node)
            n_out = sum(1 for c in genome.connections.values() if c[3] and c[0] == hovered_node)
            if ntype == NODE_INPUT and hovered_node < len(INPUT_NAMES):
                tip = f"{INPUT_NAMES[hovered_node]}  ({n_out} outgoing)"
            elif ntype == NODE_OUTPUT:
                idx = hovered_node - NUM_INPUTS
                name = OUTPUT_NAMES[idx] if 0 <= idx < len(OUTPUT_NAMES) else f"out{idx}"
                tip = f"{name}  ({n_in} incoming)"
            else:
                tip = f"Hidden #{hovered_node}  ({n_in} in, {n_out} out)"
            tip_surf = self.font_med.render(tip, True, (230, 230, 230))
            tip_bg = pygame.Rect(mx + 12, my - 10, tip_surf.get_width() + 8, 22)
            pygame.draw.rect(self.screen, (20, 22, 30), tip_bg)
            pygame.draw.rect(self.screen, (80, 80, 100), tip_bg, 1)
            self.screen.blit(tip_surf, (mx + 16, my - 8))

    @staticmethod
    def _draw_bezier(surface, p1, p2, color, thickness):
        """Draw a cubic Bézier curve between two points."""
        x1, y1 = p1
        x2, y2 = p2
        # Control points: horizontal offset for smooth S-curves
        dx = abs(x2 - x1) * 0.4
        cx1, cy1 = x1 + dx, y1
        cx2, cy2 = x2 - dx, y2
        # Sample curve
        steps = max(8, min(30, int(math.hypot(x2 - x1, y2 - y1) / 20)))
        points = []
        for i in range(steps + 1):
            t = i / steps
            u = 1 - t
            bx = u*u*u*x1 + 3*u*u*t*cx1 + 3*u*t*t*cx2 + t*t*t*x2
            by = u*u*u*y1 + 3*u*u*t*cy1 + 3*u*t*t*cy2 + t*t*t*y2
            points.append((int(bx), int(by)))
        if len(points) >= 2:
            pygame.draw.lines(surface, color, False, points, thickness)

    def _draw_body_gene_bars(self, cell, x, y, w):
        """Draw body gene bars at the bottom of brain screen."""
        bg = cell.body_genome
        bar_h = 12
        label_w = 80
        bar_w = (w - label_w) // NUM_BODY_GENES * NUM_BODY_GENES

        for i in range(NUM_BODY_GENES):
            lo, hi = BODY_GENE_RANGES[i]
            val = bg.genes[i]
            ratio = (val - lo) / (hi - lo) if hi > lo else 0.5

            bx = x + label_w + int(i * (bar_w / NUM_BODY_GENES))
            bw = int(bar_w / NUM_BODY_GENES) - 2

            # Label (only first row)
            if i < 5:
                ly = y - 14
                label = self.font_small.render(BODY_GENE_NAMES[i][:6], True, (120, 130, 140))
                self.screen.blit(label, (bx, ly))
            else:
                ly = y + bar_h + 2
                label = self.font_small.render(BODY_GENE_NAMES[i][:6], True, (120, 130, 140))
                self.screen.blit(label, (bx, ly))

            # Bar
            pygame.draw.rect(self.screen, (30, 35, 45), (bx, y, bw, bar_h))
            fill_w = int(ratio * bw)
            bar_color = (80 + int(ratio * 120), 150 - int(ratio * 50), 80)
            pygame.draw.rect(self.screen, bar_color, (bx, y, fill_w, bar_h))
            pygame.draw.rect(self.screen, (60, 65, 75), (bx, y, bw, bar_h), 1)

    # ──────── LINEAGE SCREEN (F8) ────────

    def draw_lineage_screen(self, world):
        self.screen.fill((10, 12, 18))
        title = self.font_title.render("Lineage (F8)", True, (200, 210, 220))
        self.screen.blit(title, (20, 10))

        cell = self.selected_cell
        if not cell:
            msg = self.font_med.render("Select a cell to view its lineage", True, (120, 130, 140))
            self.screen.blit(msg, (20, 50))
            return

        # Build family tree (ancestors + descendants)
        family = self._collect_family(cell.id, world)
        if not family:
            return

        # Layout: generations on Y, siblings on X
        gen_groups = defaultdict(list)
        for cid in family:
            gen = world.lineage.get(cid, {}).get('generation', 0)
            gen_groups[gen].append(cid)

        if not gen_groups:
            return

        min_gen = min(gen_groups.keys())
        max_gen = max(gen_groups.keys())
        gen_range = max(max_gen - min_gen, 1)

        # Draw
        node_pos = {}
        y_start = 60
        y_end = self.screen_h - 40
        y_step = (y_end - y_start) / (gen_range + 1)

        for gen, members in gen_groups.items():
            gy = y_start + int((gen - min_gen) * y_step)
            n = len(members)
            x_step = (self.screen_w - 100) / (n + 1)
            for i, cid in enumerate(members):
                nx = 50 + int((i + 1) * x_step)
                node_pos[cid] = (nx, gy)

        # Draw connections (parent → child)
        for cid in family:
            info = world.lineage.get(cid)
            if not info:
                continue
            if cid in node_pos and info.get('parent_id') in node_pos:
                p1 = node_pos[info['parent_id']]
                p2 = node_pos[cid]
                pygame.draw.line(self.screen, (50, 55, 65), p1, p2, 1)

        # Draw nodes
        for cid, pos in node_pos.items():
            info = world.lineage.get(cid, {})
            alive = any(c.id == cid and c.alive for c in world.cells)
            is_selected = cell.id == cid

            if is_selected:
                color = (255, 255, 100)
                r = 8
            elif alive:
                color = (100, 200, 100)
                r = 6
            else:
                color = (100, 80, 80)
                r = 5

            pygame.draw.circle(self.screen, color, pos, r)
            # ID label
            label = self.font_small.render(f"#{cid}", True, (140, 150, 160))
            self.screen.blit(label, (pos[0] - label.get_width() // 2, pos[1] + r + 2))

        instr = self.font_small.render("Press F8 to return | Click nodes to select", True, (100, 110, 120))
        self.screen.blit(instr, (20, self.screen_h - 25))

    def _collect_family(self, cell_id, world, max_depth=5):
        """Collect ancestors and descendants of a cell."""
        family = set()
        # Ancestors
        cid = cell_id
        for _ in range(max_depth):
            family.add(cid)
            info = world.lineage.get(cid)
            if not info or not info.get('parent_id'):
                break
            cid = info['parent_id']

        # Descendants (BFS)
        queue = [cell_id]
        depth = 0
        while queue and depth < max_depth:
            next_queue = []
            for cid in queue:
                family.add(cid)
                info = world.lineage.get(cid)
                if info and info.get('children'):
                    next_queue.extend(info['children'])
            queue = next_queue
            depth += 1

        return family

    # ──────── TAB BAR ────────

    def draw_tab_bar(self, current_screen):
        tabs = ["Sim (F5)", "Species (F6)", "Brain (F7)", "Lineage (F8)"]
        tw = 140
        th = 28
        tx = 10
        ty = self.screen_h - th - 2

        for i, label in enumerate(tabs):
            rect = (tx + i * (tw + 4), ty, tw, th)
            active = (i == current_screen)
            bg = (40, 45, 55) if active else (20, 22, 28)
            pygame.draw.rect(self.screen, bg, rect)
            pygame.draw.rect(self.screen, (60, 65, 75), rect, 1)
            color = (220, 220, 220) if active else (120, 130, 140)
            surf = self.font_small.render(label, True, color)
            self.screen.blit(surf, (rect[0] + 8, rect[1] + 6))

    def tab_click(self, pos):
        tw = 140
        th = 28
        tx = 10
        ty = self.screen_h - th - 2
        for i in range(4):
            rect = pygame.Rect(tx + i * (tw + 4), ty, tw, th)
            if rect.collidepoint(pos):
                return i
        return None


# ═══════════════════════════════════════════════════════════════
#  PRE-TRAINING ARENA
# ═══════════════════════════════════════════════════════════════

class PreTrainingArena:
    """Isolated mini-world for training cells before injecting into main sim.
    Supports accelerated simulation (headless, 10-50x speed)."""

    # Training scenarios
    SCENARIO_FOOD = 0       # Lots of food, no threats
    SCENARIO_FLEE = 1       # Auto-predator, little food
    SCENARIO_HUNT = 2       # Auto-prey, no plant food
    SCENARIO_FREE = 3       # Normal mini environment

    SCENARIO_NAMES = ["Food Search", "Flee Training", "Hunt Training", "Free Evolution"]

    def __init__(self, width=400, height=400, scenario=0):
        self.width = width
        self.height = height
        self.scenario = scenario
        self.world = World(width, height, boot_camp='arena')
        self.world.cells = []
        self.world.foods = []
        self.world.carcasses = []

        # Hall of fame: best genome ever seen per role (for respawn on extinction)
        self._best_ever = None   # (fitness, BodyGenome, NeatGenome)

        self.running = False
        self.tick_count = 0
        self.speed_mult = 10  # ticks per frame
        self.fitness_history = []  # (tick, avg_fitness)

        # Reduce oases to fit small world
        self.world.oases = [(width // 2, height // 2, 0)]
        self.world.water_sources = [(width // 4, height // 4)]
        self.world.obstacles = []
        self.world.shelters = []

        self._setup_scenario()

    def _setup_scenario(self):
        if self.scenario == self.SCENARIO_FOOD:
            # Lots of food (dense for directional eating training)
            for _ in range(80):
                fx = random.randint(20, self.width - 20)
                fy = random.randint(20, self.height - 20)
                self.world.foods.append(Food(fx, fy, FOOD_ENERGY, 0))
        elif self.scenario == self.SCENARIO_FLEE:
            # Sparse food + auto-predators
            for _ in range(10):
                fx = random.randint(20, self.width - 20)
                fy = random.randint(20, self.height - 20)
                self.world.foods.append(Food(fx, fy, FOOD_ENERGY, 0))
            # Spawn 2 fixed predator cells (big, carnivore)
            for _ in range(2):
                genes = list(BODY_GENE_RANGES[i][1] if i == BG_SIZE or i == BG_DIET
                             else random.uniform(*BODY_GENE_RANGES[i])
                             for i in range(NUM_BODY_GENES))
                genes[BG_SIZE] = 20.0
                genes[BG_DIET] = 0.9
                bg = BodyGenome(genes)
                pred = Cell(random.randint(50, self.width - 50),
                            random.randint(50, self.height - 50), bg)
                pred.energy = 9999
                self.world.cells.append(pred)
        elif self.scenario == self.SCENARIO_HUNT:
            # Auto-prey start STATIONARY (difficulty increases over time)
            for _ in range(8):
                self._spawn_hunt_prey(difficulty=0.0)
            # Scatter many carcasses (easy first meals while learning to hunt)
            for _ in range(15):
                cx = random.randint(30, self.width - 30)
                cy = random.randint(30, self.height - 30)
                self.world.carcasses.append(Carcass(cx, cy, FOOD_ENERGY * 3, 6.0))
        else:
            # Free: normal mini world
            for _ in range(20):
                fx = random.randint(20, self.width - 20)
                fy = random.randint(20, self.height - 20)
                self.world.foods.append(Food(fx, fy, FOOD_ENERGY, 0))

    @property
    def hunt_difficulty(self):
        """Hunt difficulty ramps up over time: 0.0 (stationary) → 1.0 (full speed).
        Phase 1 (0-5000 ticks): stationary prey (difficulty 0)
        Phase 2 (5000-20000): slow prey (0→0.5)
        Phase 3 (20000-50000): medium prey (0.5→0.8)
        Phase 4 (50000+): fast prey (0.8→1.0)"""
        t = self.tick_count
        if t < 5000:
            return 0.0
        elif t < 20000:
            return 0.5 * (t - 5000) / 15000
        elif t < 50000:
            return 0.5 + 0.3 * (t - 20000) / 30000
        else:
            return min(1.0, 0.8 + 0.2 * (t - 50000) / 50000)

    def _spawn_hunt_prey(self, difficulty=None):
        """Spawn a prey cell with speed based on difficulty (0=stationary, 1=full speed)."""
        if difficulty is None:
            difficulty = self.hunt_difficulty
        genes = [random.uniform(*BODY_GENE_RANGES[i]) for i in range(NUM_BODY_GENES)]
        genes[BG_SIZE] = random.uniform(4.0, 8.0)
        genes[BG_DIET] = 0.1
        # Scale prey speed with difficulty
        genes[BG_CILIA_POWER] = difficulty * random.uniform(1.0, 2.5)
        genes[BG_NUM_CILIA] = max(1.0, difficulty * random.uniform(2.0, 5.0))
        genes[BG_TURN_RATE] = 0.02 + difficulty * random.uniform(0.0, 0.1)
        bg = BodyGenome(genes)
        prey = Cell(random.randint(50, self.width - 50),
                    random.randint(50, self.height - 50), bg)
        prey.energy = 9999
        self.world.cells.append(prey)

    def add_cell(self, body_genome=None, neat_genome=None):
        """Add a trainee cell to the arena. Auto-seeds genes by scenario."""
        x = random.randint(50, self.width - 50)
        y = random.randint(50, self.height - 50)
        # Scenario-specific gene seeding (if no genome provided)
        if body_genome is None:
            body_genome = self._make_scenario_body()
        if neat_genome is None:
            neat_genome = self._make_scenario_brain()
        cell = Cell(x, y, body_genome, neat_genome)
        # Hunt trainees get extra starting energy (more time to learn)
        if self.scenario == self.SCENARIO_HUNT and body_genome.diet > 0.5:
            cell.energy = cell.body_genome.energy_cap * 0.8
        else:
            cell.energy = cell.body_genome.repro_threshold * 0.9
        self.world.cells.append(cell)
        self.world._record_lineage(cell)
        return cell

    def _make_scenario_body(self):
        """Create body genome biased toward the current scenario."""
        bg = BodyGenome()
        if self.scenario == self.SCENARIO_FOOD:
            # Herbivore: small-medium, wide mouth, no horns
            bg.genes[BG_DIET] = random.uniform(0.0, 0.25)
            bg.genes[BG_MOUTH_SIZE] = random.uniform(1.5, 2.5)
            bg.genes[BG_NUM_HORNS] = 0.0
            bg.genes[BG_SIZE] = random.uniform(4.0, 12.0)
        elif self.scenario == self.SCENARIO_FLEE:
            # Small fast herbivore: high cilia, small size, low diet
            bg.genes[BG_DIET] = random.uniform(0.0, 0.2)
            bg.genes[BG_SIZE] = random.uniform(3.0, 8.0)
            bg.genes[BG_NUM_CILIA] = random.uniform(3.0, 6.0)
            bg.genes[BG_CILIA_POWER] = random.uniform(1.5, 3.0)
            bg.genes[BG_TURN_RATE] = random.uniform(0.08, 0.15)
        elif self.scenario == self.SCENARIO_HUNT:
            # Carnivore: medium size (not too big = less drain), horned, fast
            bg.genes[BG_DIET] = random.uniform(0.75, 1.0)
            bg.genes[BG_SIZE] = random.uniform(7.0, 14.0)
            bg.genes[BG_NUM_HORNS] = random.uniform(1.0, 2.0)
            bg.genes[BG_HORN_SIZE] = random.uniform(1.5, 3.5)
            bg.genes[BG_HORN_SPREAD] = random.uniform(0.0, 0.3)
            bg.genes[BG_MOUTH_SIZE] = random.uniform(1.5, 2.5)  # wide mouth = easier to eat carcass
            bg.genes[BG_CILIA_POWER] = random.uniform(1.5, 3.0)
            bg.genes[BG_TURN_RATE] = random.uniform(0.08, 0.15)
            bg.genes[BG_VISION_RANGE] = random.uniform(150.0, 350.0)
        # Free evolution: fully random (default BodyGenome)
        return bg

    def _make_scenario_brain(self):
        """Create NEAT genome with scenario-specific seed connections."""
        ng = NeatGenome.create_sparse()
        if self.scenario == self.SCENARIO_HUNT:
            # Extra attack bias: prey proximity → attack output
            tracker = get_innovation_tracker()
            hunt_seeds = [
                (IN_PREY_DIST, NUM_INPUTS + OUT_ATTACK, random.uniform(1.5, 3.0)),
                (IN_PREY_SIN, NUM_INPUTS + OUT_CILIA_STEER_0, random.uniform(1.0, 2.0)),
                (IN_PREY_SIN, NUM_INPUTS + OUT_CILIA_STEER_1, random.uniform(0.5, 1.2)),
                (IN_PREY_DIST, NUM_INPUTS + OUT_CILIA_POWER_0, random.uniform(1.0, 2.0)),
                (IN_PREY_DIST, NUM_INPUTS + OUT_CILIA_POWER_1, random.uniform(0.8, 1.5)),
                (IN_PREY_COS, NUM_INPUTS + OUT_CILIA_POWER_0, random.uniform(0.5, 1.5)),
            ]
            for in_n, o_node, w in hunt_seeds:
                innov = tracker.get_innovation(in_n, o_node)
                ng.connections[innov] = (in_n, o_node, w, True)
            ng.invalidate_cache()
        elif self.scenario == self.SCENARIO_FLEE:
            # Extra flee bias: threat proximity → flee + reverse thrust
            tracker = get_innovation_tracker()
            flee_seeds = [
                # Threat → boost cilia (run away faster)
                (IN_THREAT_DIST, NUM_INPUTS + OUT_CILIA_POWER_0, random.uniform(1.0, 2.5)),
                (IN_THREAT_DIST, NUM_INPUTS + OUT_CILIA_POWER_1, random.uniform(0.8, 2.0)),
                # Threat angle → steer away (negative = turn opposite direction)
                (IN_THREAT_SIN, NUM_INPUTS + OUT_CILIA_STEER_0, random.uniform(-2.0, -1.0)),
            ]
            for in_n, o_node, w in flee_seeds:
                innov = tracker.get_innovation(in_n, o_node)
                ng.connections[innov] = (in_n, o_node, w, True)
            ng.invalidate_cache()
        return ng

    def run_ticks(self, n):
        """Run n ticks of accelerated simulation (headless)."""
        for _ in range(n):
            self.world.update()
            self.tick_count += 1
            # Regrow food/carcasses for training scenarios
            if self.tick_count % 15 == 0:
                if self.scenario in (self.SCENARIO_FOOD, self.SCENARIO_FREE):
                    while len(self.world.foods) < 50:
                        fx = random.randint(20, self.width - 20)
                        fy = random.randint(20, self.height - 20)
                        self.world.foods.append(Food(fx, fy, FOOD_ENERGY, 0))
                    # Track best-ever and respawn on extinction
                    living = [c for c in self.world.cells if c.alive]
                    for c in living:
                        if self._best_ever is None or c.fitness > self._best_ever[0]:
                            self._best_ever = (c.fitness, BodyGenome(c.body_genome.genes.copy()),
                                               NeatGenome.from_dict(c.neat_genome.to_dict()))
                    if len(living) == 0 and self._best_ever:
                        for i in range(5):
                            bg = BodyGenome(self._best_ever[1].genes.copy())
                            ng = NeatGenome.from_dict(self._best_ever[2].to_dict())
                            if i > 0:
                                bg.mutate(); ng.mutate()
                            cell = Cell(random.randint(50, self.width - 50),
                                        random.randint(50, self.height - 50), bg, ng)
                            cell.energy = cell.body_genome.repro_threshold * 0.9
                            self.world.cells.append(cell)
                elif self.scenario == self.SCENARIO_HUNT:
                    # Keep carcasses available (easy meat for learning)
                    while len(self.world.carcasses) < 8:
                        cx = random.randint(30, self.width - 30)
                        cy = random.randint(30, self.height - 30)
                        self.world.carcasses.append(Carcass(cx, cy, FOOD_ENERGY * 2, 6.0))
                    # Track best-ever predator
                    predators = [c for c in self.world.cells if c.alive and c.body_genome.diet >= 0.5]
                    for c in predators:
                        if self._best_ever is None or c.fitness > self._best_ever[0]:
                            self._best_ever = (
                                c.fitness,
                                BodyGenome(c.body_genome.genes.copy()),
                                NeatGenome.from_dict(c.neat_genome.to_dict()),
                            )
                    # Cap predator population (max 30) — kill weakest
                    if len(predators) > 30:
                        excess = sorted(predators, key=lambda c: c.fitness)
                        for c in excess[:len(predators) - 30]:
                            c.hp = 0
                    # Predator extinction → respawn from best-ever (+ mutated variants)
                    if len(predators) == 0:
                        for i in range(5):
                            if self._best_ever:
                                bg = BodyGenome(self._best_ever[1].genes.copy())
                                ng = NeatGenome.from_dict(self._best_ever[2].to_dict())
                                if i > 0:  # first is exact clone, rest mutated
                                    bg.mutate()
                                    ng.mutate()
                            else:
                                bg = self._make_scenario_body()
                                ng = self._make_scenario_brain()
                            x = random.randint(50, self.width - 50)
                            y = random.randint(50, self.height - 50)
                            cell = Cell(x, y, bg, ng)
                            cell.energy = cell.body_genome.energy_cap * 0.8
                            self.world.cells.append(cell)
                    # Cap non-predator population (max 20) and respawn auto-prey (min 5)
                    non_pred = [c for c in self.world.cells if c.alive and c.body_genome.diet < 0.5]
                    if len(non_pred) > 20:
                        excess = sorted(non_pred, key=lambda c: c.energy)
                        for c in excess[:len(non_pred) - 20]:
                            c.hp = 0
                    auto_prey = sum(1 for c in non_pred if c.energy > 5000)
                    while auto_prey < 5 and len(non_pred) < 20:
                        self._spawn_hunt_prey()
                        auto_prey += 1
                elif self.scenario == self.SCENARIO_FLEE:
                    # Keep some food for flee trainees
                    while len(self.world.foods) < 10:
                        fx = random.randint(20, self.width - 20)
                        fy = random.randint(20, self.height - 20)
                        self.world.foods.append(Food(fx, fy, FOOD_ENERGY, 0))
            # Record fitness
            if self.tick_count % 50 == 0:
                living = self.world.living_cells()
                if living:
                    avg = sum(c.fitness for c in living) / len(living)
                    self.fitness_history.append((self.tick_count, avg))

    def get_best_cell(self):
        """Return the best performing living cell."""
        living = self.world.living_cells()
        if not living:
            return None
        return max(living, key=lambda c: c.fitness)

    def get_best_genome_pair(self):
        """Return (BodyGenome, NeatGenome) of the best cell for export."""
        best = self.get_best_cell()
        if not best:
            return None, None
        # Deep copy genomes
        bg = BodyGenome(best.body_genome.genes.copy())
        ng_dict = best.neat_genome.to_dict()
        ng = NeatGenome.from_dict(ng_dict)
        return bg, ng

    def export_best_json(self):
        """Export best cell's genomes to JSON string."""
        bg, ng = self.get_best_genome_pair()
        if bg is None:
            return None
        return json.dumps({
            'body_genome': bg.to_dict(),
            'neat_genome': ng.to_dict(),
            'fitness': self.get_best_cell().fitness,
            'training_ticks': self.tick_count,
            'scenario': self.SCENARIO_NAMES[self.scenario],
        })

    def draw_full(self, renderer, screen, x, y, w, h):
        """Draw the arena using the full Renderer pipeline into a region."""
        # Save renderer state
        old_cam_x, old_cam_y = renderer.cam_x, renderer.cam_y
        old_zoom = renderer.zoom
        old_screen = renderer.screen
        old_sw, old_sh = renderer.screen_w, renderer.screen_h
        old_ww, old_wh = renderer.world_w, renderer.world_h

        # Create a subsurface for the arena viewport
        surf = pygame.Surface((w, h))
        renderer.screen = surf
        renderer.screen_w = w
        renderer.screen_h = h
        renderer.world_w = self.width
        renderer.world_h = self.height
        renderer.cam_x = 0
        renderer.cam_y = 0
        renderer.zoom = min(w / self.width, h / self.height)

        # Draw world contents
        renderer.draw(self.world, False, 1)

        # Restore renderer state
        renderer.screen = old_screen
        renderer.screen_w = old_sw
        renderer.screen_h = old_sh
        renderer.world_w = old_ww
        renderer.world_h = old_wh
        renderer.cam_x = old_cam_x
        renderer.cam_y = old_cam_y
        renderer.zoom = old_zoom

        # Blit to actual screen + border
        screen.blit(surf, (x, y))
        pygame.draw.rect(screen, (80, 120, 200), (x, y, w, h), 2)

        # Title overlay
        font = pygame.font.SysFont("consolas", 12)
        title = font.render(
            f"Arena: {self.SCENARIO_NAMES[self.scenario]} | "
            f"Tick: {self.tick_count} | "
            f"Pop: {self.world.living_count()}",
            True, (150, 170, 200)
        )
        # Semi-transparent bg for title
        tbg = pygame.Surface((title.get_width() + 8, title.get_height() + 4), pygame.SRCALPHA)
        tbg.fill((10, 12, 18, 180))
        screen.blit(tbg, (x + 2, y + 2))
        screen.blit(title, (x + 6, y + 4))

        # Fitness sparkline at bottom
        if len(self.fitness_history) > 1:
            fh = self.fitness_history[-min(50, len(self.fitness_history)):]
            max_f = max(f for _, f in fh) or 1.0
            points = []
            spark_w = w - 20
            spark_h = 20
            spark_y = y + h - 28
            for i, (_, f) in enumerate(fh):
                px = x + 10 + int(i / len(fh) * spark_w)
                py = spark_y + spark_h - int(f / max_f * spark_h)
                points.append((px, py))
            if len(points) > 1:
                pygame.draw.lines(screen, (100, 200, 100), False, points, 1)


# ═══════════════════════════════════════════════════════════════
#  SETTINGS MENU
# ═══════════════════════════════════════════════════════════════

class SettingsMenu:
    """In-game settings overlay."""

    def __init__(self, screen):
        self.screen = screen
        self.visible = False
        self.font = pygame.font.SysFont("consolas", 14)
        self.font_title = pygame.font.SysFont("consolas", 18, bold=True)

        self.settings = {
            'compat_threshold': NEAT_COMPAT_THRESHOLD,
            'add_conn_rate': NEAT_ADD_CONN_RATE,
            'add_node_rate': NEAT_ADD_NODE_RATE,
            'weight_mutate_rate': NEAT_WEIGHT_MUTATE_RATE,
            'brain_node_cost': BRAIN_NODE_COST,
            'brain_conn_cost': BRAIN_CONN_COST,
        }

    def toggle(self):
        self.visible = not self.visible

    def draw(self):
        if not self.visible:
            return

        w, h = 350, 300
        x = (self.screen.get_width() - w) // 2
        y = (self.screen.get_height() - h) // 2

        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill((15, 18, 25, 230))
        pygame.draw.rect(panel, (60, 65, 75), (0, 0, w, h), 2)
        self.screen.blit(panel, (x, y))

        ty = y + 10
        title = self.font_title.render("Settings (ESC to close)", True, (200, 210, 220))
        self.screen.blit(title, (x + 10, ty))
        ty += 30

        for key, val in self.settings.items():
            text = f"{key}: {val:.4f}"
            surf = self.font.render(text, True, (170, 180, 190))
            self.screen.blit(surf, (x + 15, ty))
            ty += 22

    def apply_to_globals(self):
        """Apply current settings to global constants."""
        global NEAT_COMPAT_THRESHOLD, NEAT_ADD_CONN_RATE, NEAT_ADD_NODE_RATE
        global NEAT_WEIGHT_MUTATE_RATE, BRAIN_NODE_COST, BRAIN_CONN_COST
        NEAT_COMPAT_THRESHOLD = self.settings['compat_threshold']
        NEAT_ADD_CONN_RATE = self.settings['add_conn_rate']
        NEAT_ADD_NODE_RATE = self.settings['add_node_rate']
        NEAT_WEIGHT_MUTATE_RATE = self.settings['weight_mutate_rate']
        BRAIN_NODE_COST = self.settings['brain_node_cost']
        BRAIN_CONN_COST = self.settings['brain_conn_cost']


# ═══════════════════════════════════════════════════════════════
#  GAME
# ═══════════════════════════════════════════════════════════════

class Game:
    """Main game loop and event handling."""

    def __init__(self):
        pygame.init()

        # Screen setup (fullscreen)
        info = pygame.display.Info()
        self.screen_w = info.current_w
        self.screen_h = info.current_h
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h), pygame.FULLSCREEN)
        pygame.display.set_caption("CelleR 2 — NEAT Evolution")

        # World (2x screen size) — starts empty, user populates from F9 arena
        self.world_w = self.screen_w * 2
        self.world_h = self.screen_h * 2
        self.world = World(self.world_w, self.world_h, boot_camp=False)

        # Renderer
        self.renderer = Renderer(self.screen, self.world_w, self.world_h)

        # Settings
        self.settings_menu = SettingsMenu(self.screen)

        # Pre-training arena (F9)
        self.arena = None  # created on demand
        self.arena_visible = False

        # State
        self.running = True
        self.paused = False
        self.sim_speed = 1
        self.current_screen = 0  # 0=Sim, 1=Species, 2=Brain, 3=Lineage, 4=Arena

        # Help overlay
        self.show_help = False

        # Flash messages (brief on-screen notifications)
        self._flash_msgs = []  # [(text, expire_tick, color)]
        self._flash_tick = 0

        # Genome library directory
        self._genome_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "celler2_genomes")

        # Camera movement keys
        self.keys_held = set()

        # Clock
        self.clock = pygame.time.Clock()

        # Auto-load saved state if exists
        self._auto_load()

    def run(self):
        while self.running:
            dt = self.clock.tick(FPS)

            self._handle_events()
            self._handle_camera()

            # Simulation update
            if not self.paused:
                for _ in range(self.sim_speed):
                    self.world.update()

            # Render
            if self.current_screen == 0:
                self.renderer.draw(self.world, self.paused, self.sim_speed)
                # Arena mini-view overlay (if running)
                if self.arena and self.arena_visible:
                    self.arena.draw_full(self.renderer, self.screen, 10, self.screen_h - 220, 300, 200)
            elif self.current_screen == 1:
                self.renderer.draw_species_screen(self.world)
            elif self.current_screen == 2:
                self.renderer.draw_brain_screen(self.world)
            elif self.current_screen == 3:
                self.renderer.draw_lineage_screen(self.world)
            elif self.current_screen == 4:
                self._draw_arena_screen()

            # Tab bar (all screens)
            self.renderer.draw_tab_bar(self.current_screen)

            # Settings overlay
            self.settings_menu.draw()

            # Flash messages
            self._draw_flash_messages()

            # Help overlay (on top of everything)
            if self.show_help:
                self._draw_help_overlay()

            pygame.display.flip()

        # Auto-save on exit
        self._save_game()
        pygame.quit()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                self.keys_held.add(event.key)
                self._handle_keydown(event.key)

            elif event.type == pygame.KEYUP:
                self.keys_held.discard(event.key)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # Tab bar click
                    tab = self.renderer.tab_click(event.pos)
                    if tab is not None:
                        self.current_screen = tab
                    elif self.current_screen == 0:
                        # Minimap click
                        if not self.renderer.minimap_click(event.pos, self.world):
                            # Cell selection
                            self.renderer.handle_click(event.pos, self.world)
                elif event.button == 4:  # Scroll up
                    self.renderer.zoom_camera(1.1, *event.pos)
                elif event.button == 5:  # Scroll down
                    self.renderer.zoom_camera(0.9, *event.pos)

    # ── Flash messages ──────────────────────────────────────────

    def _flash(self, text, color=(200, 255, 200), duration=180):
        """Show a brief on-screen notification (duration in frames)."""
        self._flash_msgs.append((text, self._flash_tick + duration, color))

    def _draw_flash_messages(self):
        """Render active flash messages at top-center."""
        self._flash_tick += 1
        self._flash_msgs = [(t, exp, c) for t, exp, c in self._flash_msgs if exp > self._flash_tick]
        if not self._flash_msgs:
            return
        font = pygame.font.SysFont("consolas", 18, bold=True)
        y = 50
        for text, exp, color in self._flash_msgs:
            # Fade out in last 30 frames
            alpha = min(255, (exp - self._flash_tick) * 255 // 30)
            surf = font.render(text, True, color)
            if alpha < 255:
                surf.set_alpha(alpha)
            x = self.screen_w // 2 - surf.get_width() // 2
            self.screen.blit(surf, (x, y))
            y += 26

    # ── Genome library ───────────────────────────────────────────

    def _ensure_genome_dir(self):
        """Create genome library directory if needed."""
        os.makedirs(self._genome_dir, exist_ok=True)

    def _save_genome_to_library(self, cell, label=""):
        """Save a cell's genome pair to the library folder. Returns filename."""
        self._ensure_genome_dir()
        # Find next available number
        existing = [f for f in os.listdir(self._genome_dir) if f.endswith('.json')]
        num = len(existing) + 1
        diet = cell.body_genome.diet
        diet_tag = "herb" if diet < 0.3 else ("carn" if diet > 0.7 else "omni")
        scenario_tag = label.lower().replace(" ", "_") if label else "manual"
        fname = f"genome_{num:03d}_{diet_tag}_{scenario_tag}_f{cell.fitness:.0f}.json"
        bg = cell.body_genome
        data = {
            'body_genome': bg.to_dict(),
            'neat_genome': cell.neat_genome.to_dict(),
            'fitness': cell.fitness,
            'diet': diet,
            'diet_tag': diet_tag,
            'size': bg.size,
            'num_horns': int(round(bg.num_horns)),
            'scenario': label,
            'brain_hidden': cell.neat_genome.num_hidden(),
            'brain_conns': cell.neat_genome.num_enabled_connections(),
        }
        path = os.path.join(self._genome_dir, fname)
        with open(path, 'w') as f:
            json.dump(data, f, default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else o)
        return fname

    def _load_genomes_from_library(self):
        """Load all genome pairs from library. Returns list of (BodyGenome, NeatGenome, metadata)."""
        if not os.path.isdir(self._genome_dir):
            return []
        results = []
        for fname in sorted(os.listdir(self._genome_dir)):
            if not fname.endswith('.json'):
                continue
            path = os.path.join(self._genome_dir, fname)
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                bg = BodyGenome.from_dict(data['body_genome'])
                ng = NeatGenome.from_dict(data['neat_genome'])
                results.append((bg, ng, data.get('fitness', 0), fname))
            except Exception:
                continue
        return results

    def _count_saved_genomes(self):
        """Count genome files in library."""
        if not os.path.isdir(self._genome_dir):
            return 0
        return sum(1 for f in os.listdir(self._genome_dir) if f.endswith('.json'))

    def _clear_genome_library(self):
        """Delete all saved genomes from library."""
        if not os.path.isdir(self._genome_dir):
            return 0
        count = 0
        for fname in os.listdir(self._genome_dir):
            if fname.endswith('.json'):
                os.remove(os.path.join(self._genome_dir, fname))
                count += 1
        return count

    def _draw_help_overlay(self):
        """Draw semi-transparent help screen with all keybindings."""
        scr = self.screen
        sw, sh = self.screen_w, self.screen_h

        # Darken background
        overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        scr.blit(overlay, (0, 0))

        font_t = pygame.font.SysFont("consolas", 28, bold=True)
        font_h = pygame.font.SysFont("consolas", 18, bold=True)
        font_r = pygame.font.SysFont("consolas", 15)

        # Center panel
        pw, ph = 700, 640
        px = sw // 2 - pw // 2
        py = sh // 2 - ph // 2
        pygame.draw.rect(scr, (20, 22, 30), (px, py, pw, ph), border_radius=10)
        pygame.draw.rect(scr, (60, 70, 90), (px, py, pw, ph), 2, border_radius=10)

        # Title
        title = font_t.render("CelleR 2 — Help", True, (100, 200, 255))
        scr.blit(title, (px + pw // 2 - title.get_width() // 2, py + 15))

        y = py + 58
        col1 = px + 30
        col2 = px + pw // 2 + 10

        sections = [
            ("Screens", [
                ("F1", "Help (this screen)"),
                ("F5", "Simulation view"),
                ("F6", "Species list"),
                ("F7", "Brain / NEAT topology"),
                ("F8", "Lineage tree"),
                ("F9", "Pre-training arena"),
                ("F10", "Settings menu"),
            ]),
            ("Simulation", [
                ("Space", "Pause / Resume"),
                ("1 / 2 / 3 / 4", "Speed: 1x / 2x / 5x / 10x"),
                ("Click", "Select cell"),
                ("Tab", "Action cam (follow cell)"),
                ("Scroll", "Zoom in / out"),
                ("Arrows / WASD", "Move camera"),
                ("Delete", "Clear all cells"),
                ("Ctrl+I", "Inject saved genomes"),
                ("ESC", "Quit (autosaves)"),
            ]),
            ("Save / Load", [
                ("Ctrl+S", "Save game"),
                ("Ctrl+L", "Load game"),
                ("", "Autosave on exit"),
            ]),
            ("Arena (F9)", [
                ("1-4", "Change scenario"),
                ("+/-", "Speed up / down"),
                ("R", "Reset arena"),
                ("S", "Save best genome"),
                ("Shift+S", "Save top 5 genomes"),
                ("C", "Clear genome library"),
                ("I", "Inject best into sim"),
                ("Space", "Pause / resume"),
            ]),
        ]

        def draw_section(sx, sy, title, items):
            head = font_h.render(title, True, (100, 200, 255))
            scr.blit(head, (sx, sy))
            sy += 26
            pygame.draw.line(scr, (50, 60, 80), (sx, sy), (sx + pw // 2 - 50, sy), 1)
            sy += 6
            for key, desc in items:
                if key:
                    ks = font_r.render(key, True, (255, 220, 100))
                    scr.blit(ks, (sx + 4, sy))
                    ds = font_r.render(desc, True, (180, 185, 195))
                    scr.blit(ds, (sx + 140, sy))
                else:
                    ds = font_r.render(desc, True, (120, 125, 135))
                    scr.blit(ds, (sx + 140, sy))
                sy += 22
            return sy + 10

        # Left column: first 2 sections
        cy = y
        cy = draw_section(col1, cy, sections[0][0], sections[0][1])
        cy = draw_section(col1, cy, sections[1][0], sections[1][1])

        # Right column: last 2 sections
        cy = y
        cy = draw_section(col2, cy, sections[2][0], sections[2][1])
        cy = draw_section(col2, cy, sections[3][0], sections[3][1])

        # Footer
        foot = font_r.render("Press F1 to close", True, (90, 95, 110))
        scr.blit(foot, (px + pw // 2 - foot.get_width() // 2, py + ph - 28))

    def _handle_keydown(self, key):
        if key == pygame.K_ESCAPE:
            if self.settings_menu.visible:
                self.settings_menu.toggle()
            else:
                self.running = False

        elif key == pygame.K_F1:
            self.show_help = not self.show_help
            return

        elif key == pygame.K_SPACE:
            self.paused = not self.paused

        elif key == pygame.K_1 and self.current_screen != 4:
            self.sim_speed = 1
        elif key == pygame.K_2 and self.current_screen != 4:
            self.sim_speed = 2
        elif key == pygame.K_3 and self.current_screen != 4:
            self.sim_speed = 5
        elif key == pygame.K_4 and self.current_screen != 4:
            self.sim_speed = 10

        elif key == pygame.K_F5:
            self.current_screen = 0
        elif key == pygame.K_F6:
            self.current_screen = 1
        elif key == pygame.K_F7:
            self.current_screen = 2
            # Auto-select: if a cell is selected, show its brain
        elif key == pygame.K_F8:
            self.current_screen = 3

        elif key == pygame.K_F9:
            if self.current_screen == 4:
                self.current_screen = 0  # toggle back
            else:
                self.current_screen = 4
                if not self.arena:
                    self.arena = PreTrainingArena(600, 600, PreTrainingArena.SCENARIO_FOOD)
                    # Add some trainees
                    for _ in range(5):
                        self.arena.add_cell()

        elif key == pygame.K_F10:
            self.settings_menu.toggle()

        elif key == pygame.K_TAB:
            # Toggle action cam
            r = self.renderer
            if r.selected_cell and r.selected_cell.alive:
                r.action_cam = not r.action_cam
                r.action_target = r.selected_cell if r.action_cam else None

        elif key == pygame.K_s and pygame.key.get_mods() & pygame.KMOD_CTRL:
            self._save_game()
        elif key == pygame.K_l and pygame.key.get_mods() & pygame.KMOD_CTRL:
            self._load_game()

        elif key == pygame.K_i and pygame.key.get_mods() & pygame.KMOD_CTRL:
            # Inject all saved genomes from library into main simulation
            genomes = self._load_genomes_from_library()
            if not genomes:
                self._flash("No saved genomes found in library", (255, 150, 150))
            else:
                for bg, ng, fitness, fname in genomes:
                    x = random.randint(100, self.world.width - 100)
                    y = random.randint(100, self.world.height - 100)
                    cell = Cell(x, y, bg, ng)
                    cell.energy = bg.repro_threshold * 0.9
                    self.world.cells.append(cell)
                    self.world._record_lineage(cell)
                self._flash(f"Injected {len(genomes)} cells from genome library", (180, 255, 180))

        elif key == pygame.K_DELETE:
            # Clear all cells from simulation
            count = len([c for c in self.world.cells if c.alive])
            self.world.cells.clear()
            self.world.carcasses.clear()
            self.renderer.selected_cell = None
            self.renderer.action_cam = False
            self.renderer.action_target = None
            self._flash(f"Cleared simulation ({count} cells removed)", (255, 200, 100))

        # Arena-specific keys (when on arena screen)
        if self.current_screen == 4:
            self._handle_arena_key(key)

    def _handle_camera(self):
        speed = 8
        if pygame.K_LEFT in self.keys_held or pygame.K_a in self.keys_held:
            self.renderer.move_camera(-speed, 0)
        if pygame.K_RIGHT in self.keys_held or pygame.K_d in self.keys_held:
            self.renderer.move_camera(speed, 0)
        if pygame.K_UP in self.keys_held or pygame.K_w in self.keys_held:
            self.renderer.move_camera(0, -speed)
        if pygame.K_DOWN in self.keys_held or pygame.K_s in self.keys_held:
            if not (pygame.key.get_mods() & pygame.KMOD_CTRL):
                self.renderer.move_camera(0, speed)

    def _draw_arena_screen(self):
        """F9: Pre-training arena full screen."""
        self.screen.fill((8, 10, 15))
        font_t = self.renderer.font_title
        font_m = self.renderer.font_med
        font_s = self.renderer.font_small

        title = font_t.render("Pre-Training Arena (F9)", True, (200, 210, 220))
        self.screen.blit(title, (20, 10))

        if not self.arena:
            msg = font_m.render("No arena active. Press F9 to create one.", True, (120, 130, 140))
            self.screen.blit(msg, (20, 50))
            return

        # Run arena ticks (accelerated)
        if not self.paused:
            self.arena.run_ticks(self.arena.speed_mult)

        # Draw arena view (large) — full renderer
        aw = min(700, self.screen_w - 350)
        ah = min(700, self.screen_h - 100)
        ax, ay = 30, 50
        self.arena.draw_full(self.renderer, self.screen, ax, ay, aw, ah)

        # Controls panel
        px = ax + aw + 30
        py = 50
        lh = 22

        def _line(text, col=(170, 180, 190)):
            nonlocal py
            s = font_s.render(text, True, col)
            self.screen.blit(s, (px, py))
            py += lh

        _line(f"Scenario: {self.arena.SCENARIO_NAMES[self.arena.scenario]}", (140, 180, 220))
        _line(f"Tick: {self.arena.tick_count}")
        _line(f"Speed: {self.arena.speed_mult}x")
        _line(f"Population: {self.arena.world.living_count()}")
        if self.arena.scenario == PreTrainingArena.SCENARIO_HUNT:
            d = self.arena.hunt_difficulty
            phase = "Stationary" if d < 0.01 else ("Slow" if d < 0.5 else ("Medium" if d < 0.8 else "Fast"))
            _line(f"Prey difficulty: {phase} ({d:.0%})", (255, 200, 100))
        py += 10

        best = self.arena.get_best_cell()
        if best:
            _line(f"Best cell: #{best.id}", (180, 220, 140))
            _line(f"  Fitness: {best.fitness:.1f}")
            _line(f"  Energy: {best.energy:.1f}")
            _line(f"  Brain: {best.neat_genome.num_hidden()}h {best.neat_genome.num_enabled_connections()}c")
        py += 10

        saved_genomes = self._load_genomes_from_library()
        saved = len(saved_genomes)
        _line(f"Genome library: {saved} saved", (200, 180, 255) if saved else (120, 120, 140))
        if saved:
            # Count by diet type
            herbs = sum(1 for _, _, _, fn in saved_genomes if '_herb_' in fn)
            carns = sum(1 for _, _, _, fn in saved_genomes if '_carn_' in fn)
            omnis = saved - herbs - carns
            parts = []
            if herbs: parts.append(f"{herbs} herb")
            if omnis: parts.append(f"{omnis} omni")
            if carns: parts.append(f"{carns} carn")
            _line(f"  ({', '.join(parts)})", (140, 150, 160))
        py += 10

        _line("Controls:", (200, 200, 150))
        _line("  1-4: Change scenario")
        _line("  +/-: Speed up/down")
        _line("  R: Reset arena")
        _line("  S: Save best genome")
        _line("  Shift+S: Save top 5")
        _line("  C: Clear genome library")
        _line("  I: Inject best into sim")
        _line("  SPACE: Pause/resume")
        _line("  F9: Return to simulation")

        # Saved genome list
        if saved_genomes:
            py += 10
            _line("Saved genomes:", (200, 180, 255))
            py += 2
            diet_colors = {'herb': (100, 200, 100), 'omni': (220, 180, 80), 'carn': (220, 100, 100)}
            for i, (bg, ng, fitness, fname) in enumerate(saved_genomes):
                if py > self.screen_h - 30:
                    _line(f"  ... +{saved - i} more", (120, 120, 140))
                    break
                # Parse diet tag from filename
                dtag = 'omni'
                if '_herb_' in fname: dtag = 'herb'
                elif '_carn_' in fname: dtag = 'carn'
                col = diet_colors.get(dtag, (170, 180, 190))
                size_val = bg.size
                horns = int(round(bg.num_horns))
                hidden = ng.num_hidden()
                conns = ng.num_enabled_connections()
                line_text = f"  {i+1}. {dtag:4s} sz:{size_val:.0f} h:{horns} f:{fitness:.0f} [{hidden}h {conns}c]"
                _line(line_text, col)

        # Fitness history graph
        if len(self.arena.fitness_history) > 2:
            gy = ay + ah + 20
            gw, gh = aw, 80
            pygame.draw.rect(self.screen, (15, 18, 25), (ax, gy, gw, gh))
            fh = self.arena.fitness_history
            max_f = max(f for _, f in fh) or 1.0
            points = []
            for i, (_, f) in enumerate(fh):
                fpx = ax + int(i / len(fh) * gw)
                fpy = gy + gh - int(f / max_f * gh)
                points.append((fpx, fpy))
            if len(points) > 1:
                pygame.draw.lines(self.screen, (100, 200, 100), False, points, 2)
            label = font_s.render("Fitness over time", True, (120, 130, 140))
            self.screen.blit(label, (ax, gy - 16))

    def _get_saveable_arena_cells(self):
        """Get arena cells eligible for saving, filtered by scenario + sorted by fitness."""
        if not self.arena:
            return []
        living = [c for c in self.arena.world.cells if c.alive]
        scenario = self.arena.scenario
        if scenario == PreTrainingArena.SCENARIO_FOOD:
            # Food search → only herbivores (diet < 0.4)
            living = [c for c in living if c.body_genome.diet < 0.4]
        elif scenario == PreTrainingArena.SCENARIO_FLEE:
            # Flee → only herbivores (diet < 0.4), exclude auto-predators (energy=9999)
            living = [c for c in living if c.body_genome.diet < 0.4 and c.energy < 9000]
        elif scenario == PreTrainingArena.SCENARIO_HUNT:
            # Hunt → only carnivores (diet > 0.6), exclude auto-prey (energy=9999)
            living = [c for c in living if c.body_genome.diet > 0.6 and c.energy < 9000]
        # Free evolution → no filter
        living.sort(key=lambda c: c.fitness, reverse=True)
        return living

    def _handle_arena_key(self, key):
        """Handle arena-specific keys."""
        if not self.arena:
            return
        if key == pygame.K_1:
            self.arena = PreTrainingArena(600, 600, PreTrainingArena.SCENARIO_FOOD)
            for _ in range(5):
                self.arena.add_cell()
        elif key == pygame.K_2:
            self.arena = PreTrainingArena(600, 600, PreTrainingArena.SCENARIO_FLEE)
            for _ in range(5):
                self.arena.add_cell()
        elif key == pygame.K_3:
            self.arena = PreTrainingArena(600, 600, PreTrainingArena.SCENARIO_HUNT)
            for _ in range(5):
                self.arena.add_cell()  # auto-seeds carnivore genes
        elif key == pygame.K_4:
            self.arena = PreTrainingArena(600, 600, PreTrainingArena.SCENARIO_FREE)
            for _ in range(5):
                self.arena.add_cell()
        elif key == pygame.K_EQUALS or key == pygame.K_PLUS:
            self.arena.speed_mult = min(50, self.arena.speed_mult + 5)
        elif key == pygame.K_MINUS:
            self.arena.speed_mult = max(1, self.arena.speed_mult - 5)
        elif key == pygame.K_r:
            scenario = self.arena.scenario
            self.arena = PreTrainingArena(600, 600, scenario)
            for _ in range(5):
                self.arena.add_cell()
        elif key == pygame.K_i:
            # Inject best eligible cell into main simulation
            eligible = self._get_saveable_arena_cells()
            if eligible:
                best = eligible[0]
                bg = BodyGenome(best.body_genome.genes.copy())
                ng = NeatGenome.from_dict(best.neat_genome.to_dict())
                x = random.randint(100, self.world.width - 100)
                y = random.randint(100, self.world.height - 100)
                cell = Cell(x, y, bg, ng)
                cell.energy = bg.repro_threshold * 0.9
                self.world.cells.append(cell)
                self.world._record_lineage(cell)
                diet_tag = "herb" if bg.diet < 0.3 else ("carn" if bg.diet > 0.7 else "omni")
                self._flash(f"Injected {diet_tag} (fitness: {best.fitness:.1f})")
            else:
                self._flash("No eligible cells to inject", (255, 150, 150))
        elif key == pygame.K_s:
            # Save genomes to library (filtered by scenario)
            mods = pygame.key.get_mods()
            # Get eligible cells (filter out auto-prey/predators + wrong diet for scenario)
            living = self._get_saveable_arena_cells()
            if not living:
                self._flash("No eligible cells to save (wrong diet for scenario?)", (255, 150, 150))
            elif mods & pygame.KMOD_SHIFT:
                # Save top 5
                saved = 0
                for c in living[:5]:
                    self._save_genome_to_library(c, label=self.arena.SCENARIO_NAMES[self.arena.scenario])
                    saved += 1
                self._flash(f"Saved top {saved} genomes to library ({self._count_saved_genomes()} total)", (180, 255, 180))
            else:
                # Save best
                best = living[0]
                fname = self._save_genome_to_library(best, label=self.arena.SCENARIO_NAMES[self.arena.scenario])
                self._flash(f"Saved best genome: {fname} ({self._count_saved_genomes()} total)", (180, 255, 180))
        elif key == pygame.K_c:
            # Clear genome library
            count = self._clear_genome_library()
            self._flash(f"Cleared genome library ({count} files deleted)", (255, 200, 100))

    def _auto_load(self):
        """Load saved state on startup if save file exists."""
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "celler2_save.json")
        if os.path.exists(path):
            try:
                self._load_game()
            except Exception as e:
                print(f"Auto-load failed: {e} — starting fresh")

    def _save_game(self):
        """Save simulation state to JSON."""
        data = {
            'tick': self.world.tick,
            'innovation_tracker': get_innovation_tracker().to_dict(),
            'cells': [],
            'foods': [{'x': f.x, 'y': f.y, 'energy': f.energy, 'oasis_id': f.oasis_id} for f in self.world.foods],
            'oases': self.world.oases,
            'water_sources': self.world.water_sources,
            'obstacles': self.world.obstacles,
            'shelters': self.world.shelters,
        }
        for cell in self.world.cells:
            if not cell.alive:
                continue
            data['cells'].append({
                'x': cell.x, 'y': cell.y,
                'angle': cell.angle,
                'energy': cell.energy,
                'hp': cell.hp,
                'thirst': cell.thirst,
                'age': cell.age,
                'fitness': cell.fitness,
                'memory_state': cell.memory_state,
                'body_genome': cell.body_genome.to_dict(),
                'neat_genome': cell.neat_genome.to_dict(),
                'parent_id': cell.parent_id,
                'generation': cell.generation,
                'children_count': cell.children_count,
                'kills': cell.kills,
                'food_eaten': cell.food_eaten,
                'species_id': cell.species_id,
            })

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "celler2_save.json")
        with open(path, 'w') as f:
            json.dump(data, f, default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else o)

    def _load_game(self):
        """Load simulation state from JSON."""
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "celler2_save.json")
        if not os.path.exists(path):
            return

        with open(path, 'r') as f:
            data = json.load(f)

        # Restore innovation tracker
        set_innovation_tracker(InnovationTracker.from_dict(data['innovation_tracker']))

        # Restore world
        w = self.world
        w.tick = data['tick']
        w.oases = [tuple(o) for o in data['oases']]
        w.water_sources = [tuple(ws) for ws in data['water_sources']]
        w.obstacles = [tuple(o) for o in data['obstacles']]
        w.shelters = [tuple(s) for s in data['shelters']]

        # Restore food
        w.foods = []
        for fd in data['foods']:
            w.foods.append(Food(fd['x'], fd['y'], fd['energy'], fd['oasis_id']))

        # Restore cells
        global _next_cell_id
        w.cells = []
        for cd in data['cells']:
            bg = BodyGenome.from_dict(cd['body_genome'])
            ng = NeatGenome.from_dict(cd['neat_genome'])
            cell = Cell(cd['x'], cd['y'], bg, ng, cd.get('parent_id'), cd.get('generation', 0))
            cell.angle = cd['angle']
            cell.energy = cd['energy']
            cell.hp = cd['hp']
            cell.thirst = cd['thirst']
            cell.age = cd['age']
            cell.fitness = cd.get('fitness', 0.0)
            cell.memory_state = cd.get('memory_state', [0.0, 0.0])
            cell.children_count = cd.get('children_count', 0)
            cell.kills = cd.get('kills', 0)
            cell.food_eaten = cd.get('food_eaten', 0)
            cell.species_id = cd.get('species_id', -1)
            cell.juvenile = False
            w.cells.append(cell)
            _next_cell_id = max(_next_cell_id, cell.id + 1)

        # Reclassify species
        w.neat_pop.classify(w.living_cells())


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    game = Game()
    game.run()
