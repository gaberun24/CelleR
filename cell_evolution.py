#!/usr/bin/env python3
"""
Egysejtű Evolúciós Szimuláció - Ősleves Ökoszisztéma

Egysejtűek élnek egy 2D világban: táplálkoznak, mozognak, szaporodnak,
mutálódnak. Ragadozók és növényevők fejlődnek ki természetes szelekcióval.

Vezérlés:
  SPACE       - Szünet/Folytatás
  W/S         - Sebesség +/-
  Nyilak      - Kamera mozgatás
  F           - Táplálék hozzáadás
  1           - Növényevő spawn (5 db a kamera közepén)
  2           - Ragadozó spawn (3 db a kamera közepén)
  M           - Beállítások menü
  R           - Reset
  TAB         - Infó panel be/ki
  Klikk       - Kiválaszt | Scroll: Zoom
  Q/ESC       - Kilépés
"""

import pygame
import numpy as np
import random
import math
import sys
from collections import defaultdict

# --- Konstansok ---
FPS = 60
FOOD_COLOR = (50, 200, 50)
FOOD_RADIUS = 3
FOOD_ENERGY = 28
MAX_FOOD = 500
FOOD_SPAWN_RATE = 0  # Nincs random spawn! Csak gócokban jelenik meg
INITIAL_CELLS = 60
INITIAL_FOOD = 18  # Ennyi góc (oázis) indul
MIN_CELL_ENERGY = 0
REPRODUCTION_COST_RATIO = 0.45  # Szaporodáskor elveszített energia aránya
MUTATION_RATE = 0.15
MUTATION_STRENGTH = 0.12
CORPSE_ENERGY_RATIO = 0.7  # Holttestekből több energia (ragadozóknak jobb)
OVERCROWDING_RADIUS = 50    # Ezen belüli sejtek számítanak zsúfoltságnak
OVERCROWDING_THRESHOLD = 5  # Ennyi sejt felett büntetés
OVERCROWDING_PENALTY = 0.03 # Extra energia fogyás szomszédonként
SPATIAL_GRID_SIZE = 60  # Térbeli rácsméret az ütközésdetektáláshoz

# Feromon rendszer
PHEROMONE_DECAY = 0.995     # Feromon halványulás per tick
PHEROMONE_MAX_AGE = 600     # Max feromon élettartam
PHEROMONE_GRID_SIZE = 40    # Feromon rács méret

# Búvóhely / fedezék
NUM_SHELTERS = 10          # Hány búvóhely a térképen
SHELTER_RADIUS = 35        # Búvóhely sugara
AMBUSH_ENERGY_MULT = 0.1   # Lesben álló ragadozó energia szorzó (90% spórolás)

# Evés rendszer
EAT_SPEED = 1.0            # Mennyi energiát szív ki tick-enként
FOOD_MIN_ENERGY = 0.5      # Ennyi alatt eltűnik a kaja

# Szín paletta a fajokhoz
def hsl_to_rgb(h, s, l):
    """HSL -> RGB konverzió."""
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


# --- Genom ---
class Genome:
    """Egy egysejtű genetikai kódja."""

    GENE_NAMES = [
        "size",          # 0: Méret (3-20)
        "sense_range",   # 1: Érzékelési sugár (20-200)
        "attack",        # 2: Támadóerő (0-10)
        "defense",       # 3: Védekezés (0-10)
        "metabolism",    # 4: Anyagcsere hatékonyság (0.3-1.0)
        "diet",          # 5: Diéta (0=növényevő, 1=ragadozó)
        "repro_thresh",  # 6: Szaporodási energiaküszöb (50-200)
        "hue",           # 7: Szín árnyalat (0-360)
        "aggression",    # 8: Agresszivitás (0-1)
        "social",        # 9: Társaság (0-1)
        "num_cilia",     # 10: Csillók száma (1-8)
        "cilia_spread",  # 11: Csillók szórása (0=egy irány, 1=körbe)
        "cilia_power",   # 12: Egy csilló tolóereje (0.3-2.0)
        "turn_rate",     # 13: Forgási sebesség (0.02-0.2 rad/tick)
        "litter_size",   # 14: Alom méret (1-4 utód egyszerre)
    ]

    def __init__(self, genes=None):
        if genes is not None:
            self.genes = np.array(genes, dtype=float)
        else:
            self.genes = np.array([
                random.uniform(5, 12),      # size
                random.uniform(120, 250),   # sense_range (messzebbről érzékelnek)
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
                random.uniform(1, 2),       # litter_size (alom méret)
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
    def max_speed(self):
        """Max sebesség a csilló konfigurációból számolva.
        Torpedó (spread=0): gyors egyenesen, lassú irányváltás
        Medúza (spread=1): lassabb egyenesen, de bármerre tud menni azonnal
        """
        # Spread hatás: torpedó hatékonyabb egyenes vonalon
        # DE medúza sem kap nagy büntetést — csak ~25% lassabb előre
        forward_efficiency = 1.0 - self.cilia_spread * 0.25
        base_speed = self.num_cilia * self.cilia_power * forward_efficiency * 0.4
        # Ragadozók aerodinamikusabbak
        predator_boost = 1.0 + self.diet * 0.35
        return base_speed * predator_boost

    @property
    def maneuverability(self):
        """Mennyire tud oldalra is menni (spread alapján).
        0 = csak előre, 1 = bármilyen irányba teljes erővel.
        """
        return self.cilia_spread

    def cilia_positions(self):
        """Csillók pozíciói a sejt körül (szög lista), 0 = előre."""
        n = self.num_cilia
        spread = self.cilia_spread
        if n == 1:
            return [0.0]
        # spread=0: mind 0 foknál (előre), spread=1: egyenletesen körbe
        angles = []
        total_arc = spread * 2 * math.pi  # 0-tól 2*pi-ig
        if total_arc < 0.01:
            return [0.0] * n
        start = -total_arc / 2
        for i in range(n):
            angle = start + (total_arc * i / (n - 1)) if n > 1 else 0
            angles.append(angle)
        return angles

    def is_predator(self):
        return self.diet > 0.5

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

    def crossover(self, other):
        mask = np.random.random(len(self.genes)) < 0.5
        child_genes = np.where(mask, self.genes, other.genes)
        return Genome(child_genes)


# --- Feromon rendszer ---
class PheromoneMap:
    """Kémiai jelek a térképen — sejtek feromonokat hagynak."""
    TRAIL = 0       # Nyomkövetés (voltam itt)
    DANGER = 1      # Veszély (ragadozó volt itt)
    FOOD_HERE = 2   # Kaja van/volt itt
    AMBUSH = 3      # Lesállás (ragadozó figyel)
    MATE = 4        # Párkereső feromon (szaporodni akarok!)
    PREY_SCENT = 5  # Préda szag (növényevő nyom, ragadozók érzékelik)

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid_size = PHEROMONE_GRID_SIZE
        self.cols = width // self.grid_size + 1
        self.rows = height // self.grid_size + 1
        # Minden cella: {type: intensity} dict
        self.grid = [[{} for _ in range(self.cols)] for _ in range(self.rows)]

    def deposit(self, x, y, ptype, intensity=1.0):
        """Feromon lerakása egy pontban."""
        col = int(x / self.grid_size)
        row = int(y / self.grid_size)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            cell = self.grid[row][col]
            cell[ptype] = min(cell.get(ptype, 0) + intensity, 5.0)

    def read(self, x, y, ptype, radius=1):
        """Feromon erősség olvasása egy pontban (szomszédos cellákból is)."""
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
        """Feromon gradiens iránya (merre erősödik)."""
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

    def decay(self):
        """Feromonok halványulása."""
        for row in self.grid:
            for cell in row:
                to_remove = []
                for ptype in cell:
                    cell[ptype] *= PHEROMONE_DECAY
                    if cell[ptype] < 0.05:
                        to_remove.append(ptype)
                for k in to_remove:
                    del cell[k]


# --- Egysejtű ---
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
        self.angle = random.uniform(0, 2 * math.pi)  # Nézési irány
        self.angular_vel = 0.0  # Forgási sebesség
        self.vx = 0.0
        self.vy = 0.0
        self.thrust = 0.0  # Aktuális hajtóerő (0-1)
        self.desired_angle = self.angle  # Hová akar fordulni
        self.cilia_phase = random.uniform(0, 2 * math.pi)  # Csilló animáció fázis
        self.kills = 0
        self.children = 0
        self.generation = 0
        # Falka/csorda
        self.target_id = None     # Aktuális célpont (préda) id-ja
        self.alert = False        # Veszélyjelzést kapott
        self.alert_x = 0.0       # Honnan jön a veszély
        self.alert_y = 0.0
        self.pack_mates = 0       # Hány fajtárs van a közelben
        # Sprint rendszer
        self.sprinting = False
        self.sprint_energy = 100.0  # Sprint tartalék (0-100)
        self.sprint_cooldown = 0    # Hűtés visszaszámlálás
        # --- Ragadozó memória ---
        # Vadászterület: sikeres vadászatok helyszínei (x, y, intensity)
        self.hunting_spots = []       # max 5 emlék
        # Préda profil: milyen préda könnyű? (defense, speed, size, pack_mates → siker?)
        self.prey_memory = []         # max 10 emlék: (defense, speed, isolated, success)
        # Preferencia ami a memóriából tanul (kezdetben semleges)
        self.prefer_weak = 0.0        # Gyenge prédát preferál (+) vs erős (-)
        self.prefer_isolated = 0.0    # Magányost preferál (+) vs csordát (-)
        self.prefer_slow = 0.0        # Lassút preferál (+) vs gyorsat (-)
        # --- Növényevő memória ---
        # Legelők: ahol korábban kaját talált (x, y, abundance)
        self.food_spots = []          # max 5 legeltető hely
        self.last_food_x = 0.0       # Utolsó evés helye
        self.last_food_y = 0.0
        # Veszélyzónák: ahol ragadozók támadtak (x, y, danger_level)
        self.danger_zones = []        # max 5 veszélyes terület
        self.attacks_here = 0         # Hányszor támadtak ezen a helyen
        self.migrating = False        # Éppen migrál-e
        self.migrate_x = 0.0         # Migráció célpontja
        self.migrate_y = 0.0
        # --- Vezetői rendszer ---
        self.leader_id = None         # Kit követ (leader cell id)
        self.leadership = 0.0         # Vezetői pontszám (dinamikusan számolt)
        self.ticks_without_food = 0   # Hány tick óta nem evett
        self.last_eat_tick = 0        # Utolsó evés időpontja
        # --- Álcázás / lesbenállás ---
        self.hiding = False           # Búvóhelyen rejtőzik
        self.ambushing = False        # Lesben áll (ragadozó)
        self.ambush_ticks = 0         # Mennyi ideje les
        self.in_shelter = False       # Búvóhelyen belül van-e
        self.eating_target = None     # Melyik kaját eszi éppen (idx)
        self.eating_progress = 0.0    # Evés haladás
        # --- HP rendszer ---
        self.max_hp = self.genome.size ** 2 * 0.5  # Nagy = sok HP
        self.hp = self.max_hp
        self.hp_regen = 0.02 + self.genome.size * 0.003  # Lassú regen
        self.damaged_flash = 0       # Sérülés vizuális effekt
        # --- Gyerekkor ---
        self.is_child = True          # Születéskor gyerek
        self.maturity = 0.4           # 40%-ról indul a mérete
        self.mature_age = 120 + int(self.genome.size * 8)  # Nagyobbak tovább nőnek
        # --- Harc ---
        self.attacking_target = None  # Kit sebez éppen
        self.being_attacked_by = []   # Kik támadják éppen
        # --- Szaporodás ---
        self.seeking_mate = False     # Aktívan keres párt
        self.mate_target_id = None    # Kiszemelt pár id-ja
        self.mate_cooldown = 0        # Szaporodás utáni pihenő

    @property
    def current_size(self):
        """Aktuális méret (gyerekkorban kisebb)."""
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
        """Sebzés amit oszt: attack * méret-szorzó."""
        size_mult = self.current_size / 10.0  # size 10 = 1x, size 20 = 2x
        return self.genome.attack * size_mult * 0.3

    @property
    def effective_speed_mult(self):
        """Sebesség szorzó: sérülés + méret lassítás."""
        hp_mult = max(0.25, self.hp_ratio)  # Féléletnél fele sebesség
        child_bonus = 1.3 if self.is_child else 1.0  # Kölykök gyorsabbak
        return hp_mult * child_bonus

    def energy_cost_per_tick(self):
        base = 0.018 if not self.genome.is_predator() else 0.04
        size_cost = self.genome.size * 0.003
        # Csilló fenntartási költség
        cilia_cost = self.genome.num_cilia * self.genome.cilia_power * 0.006
        # Aktív mozgás költsége - pihenéskor szinte nulla
        if self.thrust < 0.1:
            move_cost = 0.001  # Pihenés: szinte ingyen
        else:
            move_cost = self.thrust * self.genome.num_cilia * self.genome.cilia_power * 0.008
        # Sprint extra költség
        if self.sprinting:
            move_cost *= 2.5
        # Lesbenállás / rejtőzés: minimális energia
        if self.ambushing or self.hiding:
            return base * AMBUSH_ENERGY_MULT + size_cost * 0.5
        # Ragadozók hatékonyabb mozgással, de éhesebb anyagcserével
        if self.genome.is_predator():
            attack_cost = self.genome.attack * 0.002
            base = 0.04
        else:
            attack_cost = self.genome.attack * 0.004
        sense_cost = self.genome.sense_range * 0.00012
        efficiency = self.genome.metabolism
        return (base + size_cost + cilia_cost + move_cost + attack_cost + sense_cost) * efficiency

    def update(self, world_w, world_h):
        if not self.alive:
            return

        self.age += 1
        self.energy -= self.energy_cost_per_tick()
        if self.mate_cooldown > 0:
            self.mate_cooldown -= 1

        if self.energy <= MIN_CELL_ENERGY:
            self.alive = False
            return

        # --- HP rendszer ---
        if self.hp <= 0:
            self.alive = False
            return
        # HP regeneráció (csak ha nem harcol és van energia)
        if not self.being_attacked_by and self.hp < self.max_hp and self.energy > 20:
            self.hp = min(self.max_hp, self.hp + self.hp_regen)
        self.being_attacked_by = []  # Reset minden tick-ben
        if self.damaged_flash > 0:
            self.damaged_flash -= 1

        # --- Gyerekkor: növekedés ---
        if self.is_child:
            growth_rate = 0.6 / self.mature_age  # Fokozatosan nő
            self.maturity = min(1.0, self.maturity + growth_rate)
            if self.age >= self.mature_age:
                self.is_child = False
                self.maturity = 1.0
                # Felnőttkor: HP max frissítés
                self.max_hp = self.genome.size ** 2 * 0.5
                self.hp = self.max_hp

        # Memória halványítás (minden 200 tickenként)
        if self.age % 200 == 0:
            self.danger_zones = [(x, y, d * 0.8) for x, y, d in self.danger_zones if d * 0.8 > 0.2]
            self.hunting_spots = [(x, y, i * 0.9) for x, y, i in self.hunting_spots if i * 0.9 > 0.1]
            self.food_spots = [(x, y, a * 0.95) for x, y, a in self.food_spots if a * 0.95 > 0.15]

        if self.age > 3000:
            self.energy -= 0.02 * (self.age - 3000) / 1000

        # --- Csilló-alapú mozgás ---
        # Forgás a kívánt irányba
        angle_diff = self.desired_angle - self.angle
        # Normalizálás [-pi, pi]-re
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
        max_turn = self.genome.turn_rate
        # Magas spread = gyorsabb fordulás (medúza szinte azonnal fordul)
        # spread=0: alap turn_rate, spread=1: 4x gyorsabb
        effective_turn = max_turn * (1 + self.genome.maneuverability * 3)
        if abs(angle_diff) < effective_turn:
            self.angle = self.desired_angle
        else:
            self.angle += effective_turn if angle_diff > 0 else -effective_turn

        # --- Sprint kezelés ---
        # Nagyobb sejtek gyorsabban merülnek ki sprintből
        sprint_drain = 1.5 + self.genome.size * 0.15  # size 5=2.25, size 15=3.75, size 25=5.25
        sprint_recharge = max(0.1, 0.6 - self.genome.size * 0.02)  # Nagyobbak lassabban töltődnek
        if self.sprint_cooldown > 0:
            self.sprint_cooldown -= 1
        if self.sprinting:
            self.sprint_energy -= sprint_drain
            if self.sprint_energy <= 0:
                self.sprinting = False
                self.sprint_cooldown = int(40 + self.genome.size * 3)  # Nagyobbak tovább pihennek
                self.sprint_energy = 0
        else:
            if self.sprint_cooldown <= 0:
                self.sprint_energy = min(100, self.sprint_energy + sprint_recharge)

        # Hajtóerő számítás a csillók konfigurációja alapján
        thrust = self.thrust
        # Ragadozók erősebben sprintelnek (vadász roham)
        if self.sprinting:
            sprint_mult = 2.2 if self.genome.is_predator() else 1.6
        else:
            sprint_mult = 1.0
        cilia_angles = self.genome.cilia_positions()
        power = self.genome.cilia_power * sprint_mult
        spread = self.genome.maneuverability  # 0=torpedó, 1=medúza

        # Két komponens keveréke:
        # 1) Torpedó erő: csillók a nézési irányba tolják (hagyományos)
        # 2) Medúza erő: csillók a KÍVÁNT irányba tolják (közvetlen oldalazás)
        # A spread arány dönti el melyik dominál

        # Torpedó komponens: minden csilló a saját szögébe tol (nézési irány + offset)
        torpedo_fx, torpedo_fy = 0.0, 0.0
        for ca in cilia_angles:
            abs_angle = self.angle + ca
            torpedo_fx += math.cos(abs_angle) * power * thrust
            torpedo_fy += math.sin(abs_angle) * power * thrust

        # Medúza komponens: csillók a kívánt irányba koordinálnak
        # Minél több csilló, annál erősebb az oldalazás
        n_cilia = self.genome.num_cilia
        medusa_force = n_cilia * power * thrust
        medusa_fx = math.cos(self.desired_angle) * medusa_force
        medusa_fy = math.sin(self.desired_angle) * medusa_force

        # Keverés: spread=0 → 100% torpedó, spread=1 → 80% medúza + 20% torpedó
        fx = torpedo_fx * (1.0 - spread * 0.8) + medusa_fx * spread * 0.8
        fy = torpedo_fy * (1.0 - spread * 0.8) + medusa_fy * spread * 0.8

        # Erő alkalmazása (nagyobb = nehezebb = lassabb gyorsulás)
        mass = self.current_size * 0.8  # Erősebb méret-lassúság
        self.vx += fx / mass
        self.vy += fy / mass

        # Súrlódás (víz ellenállás — nagyobb = több ellenállás)
        drag = 0.92 - self.current_size * 0.001  # size 20 → 0.90
        drag = max(0.85, drag)
        self.vx *= drag
        self.vy *= drag

        # Sebesség limit: sérülés + gyerekkor + sprint
        speed_mult = self.effective_speed_mult
        max_spd = self.genome.max_speed * sprint_mult * speed_mult
        spd = math.sqrt(self.vx ** 2 + self.vy ** 2)
        if spd > max_spd:
            self.vx = self.vx / spd * max_spd
            self.vy = self.vy / spd * max_spd

        # Pozíció frissítés
        self.x += self.vx
        self.y += self.vy

        # Világ szélein visszapattanás
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

        # Csilló animáció
        self.cilia_phase += 0.15 * (0.5 + thrust)

    def steer_towards(self, tx, ty, thrust=0.8):
        """Kívánt irány beállítása és hajtóerő."""
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
        # Ha nemrég evett: lassan kóborol a legelő közelében
        if self.ticks_without_food < 60:
            self.desired_angle += random.gauss(0, 0.3)
            self.thrust = 0.1 + random.random() * 0.15
        else:
            # Éhes: nagyobb szögváltozásokkal keres, határozott irányba
            # Időnként teljesen új irányt vesz (kitör a körből)
            if random.random() < 0.08:
                self.desired_angle = random.uniform(0, 2 * math.pi)
            else:
                self.desired_angle += random.gauss(0, 0.5)
            self.thrust = 0.3 + random.random() * 0.3

    def remember_hunt(self, prey, success, x, y):
        """Vadászat kimenetelének megjegyzése."""
        # Préda profil mentése
        prey_speed = prey.genome.max_speed
        prey_def = prey.genome.defense
        prey_isolated = 1.0 if prey.pack_mates < 2 else 0.0
        self.prey_memory.append((prey_def, prey_speed, prey_isolated, success))
        if len(self.prey_memory) > 10:
            self.prey_memory.pop(0)  # Legrégebbi törlése

        # Preferenciák frissítése a tapasztalatból
        if len(self.prey_memory) >= 3:
            successes = [m for m in self.prey_memory if m[3]]
            failures = [m for m in self.prey_memory if not m[3]]
            if successes:
                avg_s_def = sum(m[0] for m in successes) / len(successes)
                avg_s_spd = sum(m[1] for m in successes) / len(successes)
                avg_s_iso = sum(m[2] for m in successes) / len(successes)
                # Tanulás: a sikeres vadászatok mintáit preferálja
                self.prefer_weak += (3.0 - avg_s_def) * 0.1   # alacsony def = könnyű
                self.prefer_slow += (2.0 - avg_s_spd) * 0.1   # lassú = könnyű
                self.prefer_isolated += avg_s_iso * 0.15        # magányos = könnyű
            if failures:
                avg_f_def = sum(m[0] for m in failures) / len(failures)
                avg_f_iso = sum(m[2] for m in failures) / len(failures)
                # Kerüli ami nem működött
                self.prefer_weak += (avg_f_def - 3.0) * 0.05
                self.prefer_isolated -= (1.0 - avg_f_iso) * 0.05
            # Klippelés
            self.prefer_weak = max(-1, min(2, self.prefer_weak))
            self.prefer_isolated = max(-1, min(2, self.prefer_isolated))
            self.prefer_slow = max(-1, min(2, self.prefer_slow))

        # Vadászterület mentése ha sikeres
        if success:
            self.hunting_spots.append((x, y, 1.0))
            if len(self.hunting_spots) > 5:
                self.hunting_spots.pop(0)

    def best_hunting_spot(self):
        """Legközelebbi ismert jó vadászterület."""
        if not self.hunting_spots:
            return None
        # Legjobb (legfrissebb és legerősebb)
        best = max(self.hunting_spots, key=lambda s: s[2])
        return (best[0], best[1])

    def calc_leadership(self):
        """Vezetői pontszám: tapasztalat + túlélés + social + memória."""
        score = 0
        score += min(self.age / 500, 3.0)         # Kor (max 3)
        score += self.kills * 0.5                   # Ölések (ragadozó tapasztalat)
        score += self.children * 0.3                # Szaporodás (siker)
        score += self.genome.social * 2.0           # Társas hajlam
        score += len(self.food_spots) * 0.3         # Legelő-ismeret
        score += len(self.hunting_spots) * 0.4      # Vadászhely-ismeret
        score += len(self.danger_zones) * 0.2       # Veszélyismeret
        score += self.energy * 0.01                 # Erősebb = megbízhatóbb
        self.leadership = score
        return score

    def remember_food(self, x, y):
        """Megjegyzi hol talált kaját. Közel lévő spotokat erősíti."""
        self.last_food_x = x
        self.last_food_y = y
        # Meglévő spot közelében vagyunk?
        for i, (sx, sy, abundance) in enumerate(self.food_spots):
            if abs(sx - x) < 60 and abs(sy - y) < 60:
                # Erősítjük az emléket
                self.food_spots[i] = (sx, sy, min(abundance + 0.3, 3.0))
                return
        # Új legelő felfedezve
        self.food_spots.append((x, y, 1.0))
        if len(self.food_spots) > 5:
            # Leggyengébb törlése
            self.food_spots.sort(key=lambda s: s[2], reverse=True)
            self.food_spots.pop()

    def remember_danger(self, pred_x, pred_y):
        """Megjegyzi hol támadták meg — veszélyzóna."""
        self.attacks_here += 1
        # Meglévő zóna közelében?
        for i, (dx, dy, level) in enumerate(self.danger_zones):
            if abs(dx - pred_x) < 80 and abs(dy - pred_y) < 80:
                self.danger_zones[i] = (dx, dy, min(level + 0.5, 5.0))
                return
        self.danger_zones.append((pred_x, pred_y, 1.0))
        if len(self.danger_zones) > 5:
            self.danger_zones.sort(key=lambda z: z[2], reverse=True)
            self.danger_zones.pop()

    def is_danger_zone(self, x, y):
        """Mennyire veszélyes ez a terület (0-5)."""
        danger = 0
        for dx, dy, level in self.danger_zones:
            dist_sq = (dx - x) ** 2 + (dy - y) ** 2
            if dist_sq < 120 * 120:
                danger += level * (1.0 - math.sqrt(dist_sq) / 120)
        return danger

    def best_food_spot(self):
        """Legjobb ismert legelő — figyelembe veszi a veszélyt is."""
        if not self.food_spots:
            return None
        # Pontozás: gazdagság - veszély - közelség büntetés (ne maradjon helyben)
        best = None
        best_score = -999
        for sx, sy, abundance in self.food_spots:
            dist = math.sqrt((sx - self.x) ** 2 + (sy - self.y) ** 2)
            if dist < 20:
                continue  # Már itt van, ne ide küldjük vissza
            danger = self.is_danger_zone(sx, sy)
            # Pontszám: gazdagabb + biztonságosabb + nem túl messze
            score = abundance * 2.0 - danger * 1.5 - dist * 0.002
            if score > best_score:
                best_score = score
                best = (sx, sy)
        return best

    def should_migrate(self):
        """Eldönti kell-e migrálni: túl sok támadás vagy nincs kaja."""
        # 3+ támadás ugyanitt → migráció
        if self.attacks_here >= 3:
            return True
        # A jelenlegi hely veszélyes
        if self.is_danger_zone(self.x, self.y) > 2.0:
            return True
        return False

    def start_migration(self):
        """Migrációs célpont kiválasztása — biztonságos, gazdag legelő."""
        self.migrate_tries = getattr(self, 'migrate_tries', 0)
        target = self.best_food_spot()
        if target:
            self.migrate_x, self.migrate_y = target
            self.migrating = True
            self.attacks_here = 0
            self.migrate_tries = 0
            return True
        # Nincs ismert legelő: random irány, egyre messzebb
        angle = random.uniform(0, 2 * math.pi)
        dist = 300 + self.migrate_tries * 100  # Minden próba után messzebb
        self.migrate_x = self.x + math.cos(angle) * dist
        self.migrate_y = self.y + math.sin(angle) * dist
        self.migrating = True
        self.attacks_here = 0
        self.migrate_tries += 1
        return True

    def wants_to_mate(self):
        """Kész-e párt keresni a szaporodáshoz."""
        if self.is_child or self.mate_cooldown > 0:
            return False
        thresh = self.genome.repro_thresh
        # Nemrég evett → legelőn van → alacsonyabb küszöb
        if self.ticks_without_food < 30:
            thresh *= 0.7
        if self.age < 100 or self.energy < thresh:
            return False
        # Zsúfoltságfüggő
        if self.pack_mates > 6:
            extra = (self.pack_mates - 6) * 0.15
            return self.energy >= thresh * (1.0 + extra)
        return True

    def reproduce(self, partner=None, repro_cost=None):
        litter = self.genome.litter_size
        cost = self.energy * (repro_cost if repro_cost else REPRODUCTION_COST_RATIO)
        self.energy -= cost
        # Energia elosztása az utódok között
        per_child_energy = (cost * 0.9) / litter
        self.children += litter

        children = []
        for i in range(litter):
            if partner and random.random() < 0.3:
                child_genome = self.genome.crossover(partner.genome).mutate()
            else:
                child_genome = self.genome.mutate()

            # Utódok kör alakban szétszóródnak
            angle = (2 * math.pi * i / litter) + random.uniform(-0.3, 0.3)
            dist = self.radius * 3 + random.uniform(0, self.radius * 2)
            offset_x = math.cos(angle) * dist
            offset_y = math.sin(angle) * dist
            child = Cell(self.x + offset_x, self.y + offset_y, child_genome, per_child_energy)
            child.generation = self.generation + 1
            # Gyerek HP = kis mérethez arányos
            child.max_hp = (child.genome.size * child.maturity) ** 2 * 0.5
            child.hp = child.max_hp
            # Memória öröklés: szülő átadja a tanultakat (halványabban)
            if self.hunting_spots:
                child.hunting_spots = [(x, y, iv * 0.6) for x, y, iv in self.hunting_spots]
            child.prefer_weak = self.prefer_weak * 0.7
            child.prefer_isolated = self.prefer_isolated * 0.7
            child.prefer_slow = self.prefer_slow * 0.7
            # Növényevő: legelők + veszélyzónák öröklése
            if self.food_spots:
                child.food_spots = [(x, y, a * 0.7) for x, y, a in self.food_spots]
            if self.danger_zones:
                child.danger_zones = [(x, y, d * 0.5) for x, y, d in self.danger_zones]
            children.append(child)
        return children


# --- Térbeli rács az ütközésdetektáláshoz ---
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
        """Visszaadja az összes objektumot a megadott sugáron belül."""
        results = []
        min_gx = int((x - radius) // self.cell_size)
        max_gx = int((x + radius) // self.cell_size)
        min_gy = int((y - radius) // self.cell_size)
        max_gy = int((y + radius) // self.cell_size)
        for gx in range(min_gx, max_gx + 1):
            for gy in range(min_gy, max_gy + 1):
                results.extend(self.grid.get((gx, gy), []))
        return results


# --- Világ ---
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
        }
        # Élő beállítások (menüből módosítható)
        self.settings = {
            'food_energy': FOOD_ENERGY,
            'eat_speed': EAT_SPEED,
            'food_regrow': 0.06,            # Kaja visszanövési ráta
            'repro_cost': REPRODUCTION_COST_RATIO,
            'mutation_rate': MUTATION_RATE,
            'mutation_strength': MUTATION_STRENGTH,
        }
        self._init()

    def _init(self):
        self.cells = []
        self.food = []
        self.tick = 0
        Cell._id_counter = 0

        # Feromon térkép
        self.pheromones = PheromoneMap(self.width, self.height)

        # --- Fix kaja források (oázisok) ---
        # Ezek állandó helyek ahol a növények nőnek, mint legelők
        self.food_sources = []
        for _ in range(INITIAL_FOOD):
            cx = random.uniform(150, self.width - 150)
            cy = random.uniform(150, self.height - 150)
            richness = random.uniform(0.6, 1.0)  # Mennyire termékeny
            spread = random.uniform(60, 120)  # Nagyobb szórás, könnyebb megtalálni
            self.food_sources.append({
                'x': cx, 'y': cy,
                'richness': richness,
                'spread': spread,
                'capacity': int(25 + richness * 30),  # Max kaja ami itt nőhet
            })
        # Búvóhelyek: fele oázis mellett (stratégiai), fele random
        self.shelters = []
        for i in range(NUM_SHELTERS):
            if i < len(self.food_sources) and i < NUM_SHELTERS // 2:
                # Oázis mellé (leselkedő pozíció!)
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

        # Kezdő kaja: oázisokból
        for src in self.food_sources:
            for _ in range(src['capacity'] // 2):
                x = src['x'] + random.gauss(0, src['spread'])
                y = src['y'] + random.gauss(0, src['spread'])
                if 10 < x < self.width - 10 and 10 < y < self.height - 10:
                    food_e = self.settings['food_energy'] * random.uniform(0.4, 1.8)
                    self.food.append((x, y, food_e, False))

        # Kezdő egysejtűek
        for _ in range(INITIAL_CELLS):
            x = random.uniform(50, self.width - 50)
            y = random.uniform(50, self.height - 50)
            cell = Cell(x, y, energy=80)
            self.cells.append(cell)
            self.stats["total_born"] += 1

    def _spawn_food_cluster(self):
        """Kaja újranövesztése egy random oázisban."""
        if not self.food_sources:
            return
        src = random.choice(self.food_sources)
        # Mennyi kaja van most ebben az oázisban
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

    def update(self):
        self.tick += 1

        # Feromon halványulás
        if self.tick % 3 == 0:  # Nem minden tick-ben, CPU kímélés
            self.pheromones.decay()

        # Táplálék: oázisokban nő vissza
        if len(self.food) < MAX_FOOD:
            for src in self.food_sources:
                # Alap visszanövés: folyamatos
                if random.random() < self.settings['food_regrow'] * src['richness']:
                    nearby_food = sum(1 for f in self.food
                                     if abs(f[0] - src['x']) < src['spread'] * 2
                                     and abs(f[1] - src['y']) < src['spread'] * 2)
                    if nearby_food < src['capacity']:
                        # Burst spawn: minél kevesebb a kaja, annál gyorsabb visszanövés
                        if nearby_food < src['capacity'] * 0.3:
                            burst = random.randint(2, 4)  # Nagyon üres: gyors regenerálódás
                        elif nearby_food < src['capacity'] * 0.6:
                            burst = random.randint(1, 3)
                        else:
                            burst = 1
                        for _ in range(burst):
                            x = src['x'] + random.gauss(0, src['spread'])
                            y = src['y'] + random.gauss(0, src['spread'])
                            if 10 < x < self.width - 10 and 10 < y < self.height - 10 and len(self.food) < MAX_FOOD:
                                # Változó méretű kaja: kicsi fűszál → nagy bokor
                                base_e = self.settings['food_energy']
                                food_e = base_e * random.uniform(0.4, 1.8)
                                self.food.append((x, y, food_e, False))
                                self.pheromones.deposit(x, y, PheromoneMap.FOOD_HERE, 0.3)

        # Térbeli rácsok frissítése
        self.cell_grid.clear()
        self.food_grid.clear()
        for cell in self.cells:
            if cell.alive:
                self.cell_grid.insert(cell, cell.x, cell.y)
        for i, food_item in enumerate(self.food):
            self.food_grid.insert(i, food_item[0], food_item[1])

        # AI és mozgás
        new_cells = []
        for cell in self.cells:
            if not cell.alive:
                continue
            # Búvóhely check
            cell.in_shelter = self.is_in_shelter(cell.x, cell.y)
            # Feromon nyomok
            if self.tick % 5 == 0:
                if cell.genome.is_predator():
                    self.pheromones.deposit(cell.x, cell.y, PheromoneMap.DANGER, 0.2)
                else:
                    self.pheromones.deposit(cell.x, cell.y, PheromoneMap.TRAIL, 0.15)
                    # Növényevők préda szagot hagynak (ragadozók érzékelik)
                    self.pheromones.deposit(cell.x, cell.y, PheromoneMap.PREY_SCENT, 0.25)
                # Párkereső feromon: egyre erősebb minél régebb óta keres
                if cell.seeking_mate:
                    mate_intensity = min(0.8, 0.15 + cell.age * 0.001)
                    self.pheromones.deposit(cell.x, cell.y, PheromoneMap.MATE, mate_intensity)
            self._cell_ai(cell)

            # --- Túlzsúfoltság büntetés ---
            crowding = self.cell_grid.query(cell.x, cell.y, OVERCROWDING_RADIUS)
            crowd_count = len(crowding) - 1  # Saját magát nem számoljuk
            if crowd_count > OVERCROWDING_THRESHOLD:
                excess = crowd_count - OVERCROWDING_THRESHOLD
                cell.energy -= excess * OVERCROWDING_PENALTY

            cell.update(self.width, self.height)

            if not cell.alive:
                # Holttest → tetem (méret-arányos energia)
                corpse_energy = cell.energy * 0.5 + cell.genome.size ** 2 * 0.6
                corpse_energy = max(5, corpse_energy)
                self.food.append((cell.x, cell.y, corpse_energy, True))
                self.stats["total_died"] += 1
                continue

            # Mindenki eszik - de a fajta alapján: növényevő=növény, ragadozó=hús
            self._eat_food(cell)

            # Szaporodás: párkeresés szükséges!
            if cell.wants_to_mate():
                cell.seeking_mate = True
                # Keresünk párt a közelben — mindkettőnek késznek kell lennie
                mate_range = cell.radius * 3 + 5  # Elég közel kell lenniük
                partner = None
                nearby = self.cell_grid.query(cell.x, cell.y, mate_range)
                for other in nearby:
                    if other.id == cell.id or not other.alive:
                        continue
                    # Azonos fajta (ragadozó+ragadozó vagy növényevő+növényevő)
                    same_type = abs(other.genome.diet - cell.genome.diet) < 0.3
                    if same_type and other.wants_to_mate():
                        dx = other.x - cell.x
                        dy = other.y - cell.y
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < mate_range:
                            partner = other
                            break
                if partner:
                    # Mindketten szaporodnak, a partner is kap cooldown-t
                    children = cell.reproduce(partner, repro_cost=self.settings['repro_cost'])
                    cell.mate_cooldown = 200  # ~3 másodperc pihenő
                    partner.mate_cooldown = 200
                    cell.seeking_mate = False
                    partner.seeking_mate = False
                    partner.energy -= partner.energy * self.settings['repro_cost'] * 0.3  # Partner is ad energiát
                    for child in children:
                        child.x = np.clip(child.x, 5, self.width - 5)
                        child.y = np.clip(child.y, 5, self.height - 5)
                        new_cells.append(child)
                        self.stats["total_born"] += 1
                        self.stats["max_generation"] = max(self.stats["max_generation"], child.generation)
            else:
                cell.seeking_mate = False

        # Ragadozó támadások
        for cell in self.cells:
            if cell.alive and cell.genome.is_predator():
                self._predator_attack(cell)

        # Halottak eltávolítása, újak hozzáadása
        self.cells = [c for c in self.cells if c.alive]
        # Populáció limit - ha túl sokan vannak, csak a legfittebb újak jönnek be
        max_pop = 400
        if len(self.cells) + len(new_cells) > max_pop:
            new_cells = new_cells[:max(0, max_pop - len(self.cells))]
        self.cells.extend(new_cells)

        # Statisztikák frissítése
        self.stats["predators"] = sum(1 for c in self.cells if c.genome.is_predator())
        self.stats["herbivores"] = len(self.cells) - self.stats["predators"]

        # Auto-spawn KIKAPCSOLVA — manuálisan lehet spawnolni (1/2 gomb)

    def spawn_herbivore(self, x, y):
        """Manuális növényevő spawn."""
        g = Genome()
        g.genes[5] = random.uniform(0.0, 0.3)   # diet: növényevő
        g.genes[1] = random.uniform(150, 280)    # sense_range: jó érzékelés
        g.genes[3] = random.uniform(2, 5)        # defense
        g.genes[6] = random.uniform(35, 70)      # repro_thresh: alacsony küszöb
        g.genes[9] = random.uniform(0.3, 0.7)    # social
        g.genes[14] = random.uniform(1.5, 3)     # litter_size: több utód
        c = Cell(x, y, g, energy=100)
        self.cells.append(c)
        self.stats["total_born"] += 1

    def spawn_predator(self, x, y):
        """Manuális ragadozó spawn."""
        g = Genome()
        g.genes[5] = random.uniform(0.7, 1.0)
        g.genes[2] = random.uniform(3, 6)
        g.genes[1] = random.uniform(100, 200)
        g.genes[9] = random.uniform(0.3, 0.7)
        c = Cell(x, y, g, energy=120)
        self.cells.append(c)
        self.stats["total_born"] += 1

    def is_in_shelter(self, x, y):
        """Búvóhelyen belül van-e."""
        for s in self.shelters:
            dx = x - s['x']
            dy = y - s['y']
            if dx * dx + dy * dy < s['radius'] * s['radius']:
                return True
        return False

    def _cell_ai(self, cell):
        """Egysejtű viselkedés falka/csorda mechanikával + vezetői rendszer."""
        sense = cell.genome.sense_range
        nearby_cells = self.cell_grid.query(cell.x, cell.y, sense)

        # Éhezés számláló (első 300 tick-ben nincs éhezés, van induló energia)
        if cell.last_eat_tick > 0:
            cell.ticks_without_food = self.tick - cell.last_eat_tick
        elif cell.age > 300:
            cell.ticks_without_food = cell.age - 300
        else:
            cell.ticks_without_food = 0

        # Fajtársak és ellenségek szétválogatása + vezető keresés
        allies = []
        enemies = []
        cell.pack_mates = 0
        best_leader = None
        best_leader_score = cell.calc_leadership()  # Saját leadership

        for other in nearby_cells:
            if other.id == cell.id or not other.alive:
                continue
            dx = other.x - cell.x
            dy = other.y - cell.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > sense:
                continue
            is_same_type = abs(other.genome.diet - cell.genome.diet) < 0.3
            if is_same_type:
                allies.append((other, dist))
                cell.pack_mates += 1
                # Vezető keresés: tapasztaltabb + szociálisabb = jobb vezető
                if cell.genome.social > 0.3:
                    other_score = other.calc_leadership()
                    if other_score > best_leader_score:
                        best_leader = other
                        best_leader_score = other_score
            else:
                enemies.append((other, dist))

        # Vezető frissítés
        cell.leader_id = best_leader.id if best_leader else None

        if cell.genome.is_predator():
            self._predator_ai(cell, allies, enemies)
        else:
            self._herbivore_ai(cell, allies, enemies)

        # --- Párkeresés: ha szaporodni akar, fajtárs felé megy ---
        if cell.seeking_mate:
            found_mate = False
            if allies:
                best_mate = None
                best_mate_dist = cell.genome.sense_range
                for ally, dist in allies:
                    if ally.wants_to_mate() and dist < best_mate_dist:
                        best_mate = ally
                        best_mate_dist = dist
                        cell.mate_target_id = ally.id
                if best_mate:
                    # Látja a párt — felé megy
                    mate_thrust = 0.5 if best_mate_dist > 50 else 0.7
                    cell.steer_towards(best_mate.x, best_mate.y, mate_thrust)
                    found_mate = True
            if not found_mate:
                # Nem lát párt — szagolja a MATE feromon gradienst
                gx, gy = self.pheromones.read_gradient(cell.x, cell.y, PheromoneMap.MATE)
                if abs(gx) + abs(gy) > 0.2:
                    cell.steer_towards(cell.x + gx * 30, cell.y + gy * 30, 0.45)
                    cell.mate_target_id = None

        # --- Társas vonzás: vezető felé erősebb ---
        if cell.genome.social > 0.15 and allies:
            if best_leader and cell.leader_id:
                # Vezetőt követi (erősebben)
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
                # Ha a vezető migrál, a követő is migrál
                if best_leader.migrating and not cell.migrating and cell.genome.social > 0.3:
                    cell.migrating = True
                    cell.migrate_x = best_leader.migrate_x + random.gauss(0, 20)
                    cell.migrate_y = best_leader.migrate_y + random.gauss(0, 20)
            else:
                # Nincs vezető, átlag felé húz
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
            # Taszítás ha túl közel
            for ally, adist in allies:
                if adist < cell.radius * 2:
                    dx = cell.x - ally.x
                    dy = cell.y - ally.y
                    cell.desired_angle = math.atan2(
                        math.sin(cell.desired_angle) + dy / max(adist, 1) * 0.02,
                        math.cos(cell.desired_angle) + dx / max(adist, 1) * 0.02
                    )
                    break

            # --- Sebesség szinkron: hasonló sebességgel mozognak ---
            if cell.genome.social > 0.4 and len(allies) >= 2:
                avg_vx = sum(a.vx for a, _ in allies) / len(allies)
                avg_vy = sum(a.vy for a, _ in allies) / len(allies)
                sync = cell.genome.social * 0.1
                cell.vx += (avg_vx - cell.vx) * sync
                cell.vy += (avg_vy - cell.vy) * sync

    def _predator_ai(self, cell, allies, enemies):
        """Ragadozó AI falkavadászattal."""
        sense = cell.genome.sense_range

        # Préda keresés - saját + falkatársak célpontja
        best_prey = None
        best_score = -1

        # Megnézzük a falkatársak célpontját (közös célpont)
        pack_target_id = None
        if cell.genome.social > 0.4:
            for ally, _ in allies:
                if ally.target_id is not None:
                    pack_target_id = ally.target_id
                    break
                # Memória megosztás: falkatárs vadászhelyeinek átvétele
                if ally.hunting_spots and not cell.hunting_spots:
                    cell.hunting_spots = [(x, y, i * 0.5) for x, y, i in ally.hunting_spots[:2]]

        # --- Opportunista: van tetem (hús) a közelben? Könnyebb mint vadászni! ---
        nearby_meat = self.food_grid.query(cell.x, cell.y, sense)
        best_meat = None
        best_meat_dist = sense
        for idx in nearby_meat:
            if idx >= len(self.food):
                continue
            f = self.food[idx]
            if len(f) > 3 and f[3]:  # is_meat = True
                dx = f[0] - cell.x
                dy = f[1] - cell.y
                d = math.sqrt(dx*dx + dy*dy)
                if d < best_meat_dist:
                    best_meat = (f[0], f[1])
                    best_meat_dist = d
                    best_meat_energy = f[2]

        if best_meat:
            # Tetem van! Minek vadászni ha ingyen van?
            # Csak ha nagyon agresszív ÉS a préda közelebb van, vadászik inkább
            eat_meat = True
            if cell.genome.aggression > 0.7 and enemies:
                # Nagyon agresszív: megnézi van-e könnyű préda közelebb
                closest_prey_dist = min((d for _, d in enemies), default=sense)
                if closest_prey_dist < best_meat_dist * 0.5:
                    eat_meat = False  # Préda közelebb van, vadászik
            if eat_meat:
                thrust = 0.5 if best_meat_dist > 30 else 0.2
                cell.steer_towards(best_meat[0], best_meat[1], thrust)
                cell.target_id = None
                cell.sprinting = False
                return

        # Elérhető prédák értékelése
        all_nearby = self.cell_grid.query(cell.x, cell.y, sense)
        for other in all_nearby:
            if other.id == cell.id or not other.alive:
                continue
            # Ragadozó NEM vadászik ragadozóra
            if other.genome.is_predator():
                continue

            dx = other.x - cell.x
            dy = other.y - cell.y
            dist = max(math.sqrt(dx * dx + dy * dy), 1)

            # --- Mérlegelés: bírok-e vele? ---
            my_damage = cell.damage_per_tick
            prey_hp = other.hp
            # Hány tick kellene egyedül megölni
            ticks_to_kill = prey_hp / max(0.1, my_damage)
            # Falka segít: minden társsal gyorsabb
            pack_attackers = 1 + sum(1 for a, _ in allies if a.target_id == other.id)
            ticks_with_pack = ticks_to_kill / pack_attackers

            # Túl nagy préda egyedül? Skip!
            if ticks_to_kill > 30 and pack_attackers < 2:
                continue  # Nem éri meg egyedül
            # Gyerekek könnyebb célpont
            child_bonus = 0.3 if other.is_child else 0

            # Pontozás: közelibb + gyengébb + kevesebb HP = jobb
            score = (sense - dist) / sense
            score += (cell.genome.attack - other.genome.defense) * 0.1
            score -= ticks_with_pack * 0.02  # Gyorsabb kill = jobb
            score += child_bonus
            score += (1.0 - other.hp_ratio) * 0.5  # Sérültet preferálja!
            # Ha ez a falka célpontja, bónusz
            if pack_target_id and other.id == pack_target_id:
                score += 0.5 * cell.genome.social
            # Kisebb préda = könnyebb
            score += (cell.current_size - other.current_size) * 0.02

            # Memória-alapú preferenciák
            if cell.prefer_weak > 0:
                score += (5.0 - other.genome.defense) * cell.prefer_weak * 0.1
            if cell.prefer_isolated > 0:
                is_alone = 1.0 if other.pack_mates < 2 else -0.5
                score += is_alone * cell.prefer_isolated * 0.3
            if cell.prefer_slow > 0:
                score += (2.0 - other.genome.max_speed) * cell.prefer_slow * 0.15

            if score > best_score:
                best_prey = other
                best_score = score

        if best_prey and random.random() < cell.genome.aggression:
            # Lesből támadás! Meglepetés bónusz
            if cell.ambushing:
                cell.ambushing = False
                cell.hiding = False
                cell.ambush_ticks = 0
                cell.sprinting = True  # Azonnali sprint roham!

            cell.target_id = best_prey.id
            dx = best_prey.x - cell.x
            dy = best_prey.y - cell.y
            prey_dist = math.sqrt(dx * dx + dy * dy)

            # --- Sprint aktiválás: közel a prédához → roham! ---
            if prey_dist < sense * 0.4 and cell.sprint_energy > 30 and cell.sprint_cooldown <= 0:
                cell.sprinting = True
            elif prey_dist > sense * 0.6:
                cell.sprinting = False  # Távol van, energiát spórol

            # Energiatakarékos megközelítés: messziről lassabban, közelről roham
            if prey_dist > sense * 0.5:
                approach_thrust = 0.45  # Távoli: lassú, spórolós közelítés
            elif prey_dist > sense * 0.25:
                approach_thrust = 0.65  # Közepes: fokozódó sebesség
            else:
                approach_thrust = 0.9   # Közel: teljes gáz
            cell.steer_towards(best_prey.x, best_prey.y, approach_thrust)

            # --- Falkavadászat jelzés: célpont megosztás fajtársakkal ---
            if cell.genome.social > 0.4:
                for ally, adist in allies:
                    if adist < sense * 0.6 and ally.genome.social > 0.3:
                        if ally.target_id is None or ally.target_id != best_prey.id:
                            ally.target_id = best_prey.id  # Közös célpont

            # --- Bekerítés: falkatagok oldalról közelítenek ---
            if cell.genome.social > 0.4 and allies:
                pack_count = sum(1 for a, d in allies if a.target_id == best_prey.id)
                if pack_count > 0:
                    dist = max(prey_dist, 1)
                    side = 1 if cell.id % 2 == 0 else -1
                    offset = cell.radius * 4 * side
                    tx = best_prey.x + (-dy / dist) * offset
                    ty = best_prey.y + (dx / dist) * offset
                    cell.steer_towards(tx, ty, 0.75)
        else:
            cell.target_id = None
            cell.sprinting = False

            # --- Éhezés-alapú vándorlás ---
            # Ha régóta nem evett, egyre kétségbeesettebben keres
            hunger = cell.ticks_without_food
            is_starving = hunger > 400  # ~400 tick éhezés után vándorol

            if is_starving and not cell.migrating:
                # Életösztön: el kell menni innen, itt nincs mit enni
                cell.migrating = True
                # Célpont: ismert vadászhely vagy random irány messzire
                spot = cell.best_hunting_spot()
                if spot:
                    # Megy a legjobb vadászhelyre
                    cell.migrate_x = spot[0] + random.gauss(0, 50)
                    cell.migrate_y = spot[1] + random.gauss(0, 50)
                else:
                    # Random irány, nagy távolság — felfedezés
                    angle = random.uniform(0, 2 * math.pi)
                    cell.migrate_x = cell.x + math.cos(angle) * 500
                    cell.migrate_y = cell.y + math.sin(angle) * 500
                cell.migrate_x = max(50, min(self.width - 50, cell.migrate_x))
                cell.migrate_y = max(50, min(self.height - 50, cell.migrate_y))
                # Falkavezető: falkatársak is vándorolnak vele
                if cell.leader_id is None and cell.genome.social > 0.3:
                    for ally, adist in allies:
                        if ally.leader_id == cell.id and not ally.migrating:
                            ally.migrating = True
                            ally.migrate_x = cell.migrate_x + random.gauss(0, 30)
                            ally.migrate_y = cell.migrate_y + random.gauss(0, 30)

            if cell.migrating:
                # Vándorlás mód
                dx = cell.migrate_x - cell.x
                dy = cell.migrate_y - cell.y
                dist_to_target = math.sqrt(dx * dx + dy * dy)
                if dist_to_target < 60:
                    cell.migrating = False
                    # Ha még mindig éhes, próbál új helyet
                    if cell.ticks_without_food > 200:
                        cell.start_migration()
                else:
                    # Útközben préda szagot érez? Megáll vadászni
                    prey_smell = self.pheromones.read(cell.x, cell.y, PheromoneMap.PREY_SCENT, radius=2)
                    if prey_smell > 1.0:
                        cell.migrating = False  # Van préda a közelben!
                    else:
                        cell.steer_towards(cell.migrate_x, cell.migrate_y, 0.6)
                        if hunger > 400 and cell.sprint_energy > 30:
                            cell.sprinting = True
                        return

            # --- Lesbenállás: búvóhely mellett várakozás ---
            if cell.in_shelter and cell.energy > 30:
                cell.ambushing = True
                cell.ambush_ticks += 1
                cell.thrust = 0.0  # Teljesen mozdulatlan
                cell.hiding = True
                self.pheromones.deposit(cell.x, cell.y, PheromoneMap.AMBUSH, 0.1)
                # Max 500 tick lesben, utána megy tovább
                if cell.ambush_ticks > 500:
                    cell.ambushing = False
                    cell.ambush_ticks = 0
                return
            elif not cell.in_shelter:
                cell.ambushing = False
                cell.ambush_ticks = 0
                cell.hiding = False

            # Nincs préda: keress búvóhelyet lesbe, vadászhelyet, vagy kóborolj
            # Próbálj búvóhelyet keresni legelő közelében (oázis + shelter = csapda!)
            best_ambush = None
            best_ambush_dist = 9999
            for s in self.shelters:
                # Van préda szag a közelében? → jó lesállás
                prey_near = self.pheromones.read(s['x'], s['y'], PheromoneMap.PREY_SCENT)
                food_near = self.pheromones.read(s['x'], s['y'], PheromoneMap.FOOD_HERE)
                if prey_near > 0.3 or food_near > 0.5:
                    dx = s['x'] - cell.x
                    dy = s['y'] - cell.y
                    d = math.sqrt(dx*dx + dy*dy)
                    if d < best_ambush_dist:
                        best_ambush = s
                        best_ambush_dist = d

            if best_ambush and best_ambush_dist > 20 and random.random() < 0.4:
                # Leshelyre energiatakarékosan: lassú, feltűnésmentes közelítés
                ambush_thrust = 0.3 if best_ambush_dist > 100 else 0.2
                cell.steer_towards(best_ambush['x'], best_ambush['y'], ambush_thrust)
                return

            spot = cell.best_hunting_spot()
            if spot and random.random() < 0.6:
                dx = spot[0] - cell.x
                dy = spot[1] - cell.y
                dist_to_spot = math.sqrt(dx * dx + dy * dy)
                if dist_to_spot > 30:
                    # Vadászterületre: éhség alapú sebesség
                    hunger = cell.ticks_without_food
                    hunt_thrust = 0.35 if hunger < 200 else 0.55
                    cell.steer_towards(spot[0], spot[1], hunt_thrust)
                else:
                    cell.hunting_spots = [(x, y, i * 0.9) for x, y, i in cell.hunting_spots]
                    cell.wander()
            elif not self._seek_food_or_wander(cell):
                # --- Ragadozó szaglás: préda szag követése ---
                # Először a friss préda szagot keresi (erősebb, célzottabb)
                gx, gy = self.pheromones.read_gradient(cell.x, cell.y, PheromoneMap.PREY_SCENT)
                scent_strength = abs(gx) + abs(gy)
                if scent_strength > 0.2:
                    # Van préda szag! Éhesebb → gyorsabban követi
                    hunger = cell.ticks_without_food
                    scent_thrust = 0.35 if hunger < 150 else 0.55
                    target_x = cell.x + gx * 25
                    target_y = cell.y + gy * 25
                    cell.steer_towards(target_x, target_y, scent_thrust)
                else:
                    # Nincs préda szag — régi nyomot (TRAIL) követi halványabban
                    gx2, gy2 = self.pheromones.read_gradient(cell.x, cell.y, PheromoneMap.TRAIL)
                    if abs(gx2) + abs(gy2) > 0.3:
                        cell.steer_towards(cell.x + gx2 * 20, cell.y + gy2 * 20, 0.3)
                    else:
                        cell.thrust = 0.15

    def _herbivore_ai(self, cell, allies, enemies):
        """Növényevő AI csordavédelemmel, migrációval és veszélykerüléssel."""
        sense = cell.genome.sense_range

        # --- Álcázás: búvóhelyen rejtőzés ha veszély van ---
        if cell.in_shelter and cell.hiding:
            cell.thrust = 0.05
            # Ha nincs veszély a közelben, előjön
            if not enemies:
                danger_smell = self.pheromones.read(cell.x, cell.y, PheromoneMap.DANGER)
                if danger_smell < 0.3:
                    cell.hiding = False

        # --- Feromon szaglás: veszély szag érzékelés ---
        danger_smell = self.pheromones.read(cell.x, cell.y, PheromoneMap.DANGER, radius=2)
        ambush_smell = self.pheromones.read(cell.x, cell.y, PheromoneMap.AMBUSH, radius=2)
        if danger_smell > 1.5 or ambush_smell > 0.5:
            # Ragadozó szagot érez — menekülj a szag ellenében!
            gx, gy = self.pheromones.read_gradient(cell.x, cell.y, PheromoneMap.DANGER)
            if abs(gx) + abs(gy) > 0.2:
                cell.steer_away(cell.x + gx * 30, cell.y + gy * 30, 0.7)
                # Búvóhely keresés ha közelben van
                for s in self.shelters:
                    dx = s['x'] - cell.x
                    dy = s['y'] - cell.y
                    d = math.sqrt(dx*dx + dy*dy)
                    if d < sense * 0.5:
                        cell.steer_towards(s['x'], s['y'], 0.6)
                        cell.hiding = True
                        break
                return

        # --- Veszélyérzékelés ---
        danger = None
        danger_dist = sense + 1

        for other, dist in enemies:
            # Lesben álló ragadozót nehezebb észrevenni
            if other.genome.is_predator() and other.genome.attack > cell.genome.defense * 0.3:
                detect_dist = dist
                if other.ambushing or other.hiding:
                    detect_dist = dist * 0.5  # Fele olyan messzire érzékeli
                if detect_dist < danger_dist:
                    danger = other
                    danger_dist = dist

        # Veszélyjelzés társaktól
        if not danger and cell.alert and cell.genome.social > 0.3:
            cell.steer_away(cell.alert_x, cell.alert_y, 0.6)
            cell.alert = False
            return

        if danger and danger_dist < sense * 0.6:
            # MENEKÜLÉS + Sprint!
            cell.steer_away(danger.x, danger.y, 0.95)
            if danger_dist < sense * 0.35 and cell.sprint_energy > 20 and cell.sprint_cooldown <= 0:
                cell.sprinting = True  # Pánik sprint!

            # Veszélyzóna megjegyzése!
            cell.remember_danger(danger.x, danger.y)

            # --- Veszélyjelzés a csordának ---
            if cell.genome.social > 0.3:
                for ally, dist in allies:
                    if dist < sense * 0.6 and ally.genome.social > 0.2:
                        ally.alert = True
                        ally.alert_x = danger.x
                        ally.alert_y = danger.y
                        # Veszélyzóna megosztás
                        ally.remember_danger(danger.x, danger.y)

            # --- Csordavédelem: ha elég társunk van, bátrabban viselkedünk ---
            if cell.pack_mates >= 3 and cell.genome.social > 0.5:
                if danger_dist > sense * 0.3:
                    self._seek_food_or_wander(cell)
                    return

            # --- Migrációs döntés: túl sokat zaklatnak itt? ---
            if cell.should_migrate() and not cell.migrating:
                cell.start_migration()
                # Csordatársak is migrálnak!
                if cell.genome.social > 0.4:
                    for ally, dist in allies:
                        if ally.genome.social > 0.3 and not ally.migrating:
                            ally.migrate_x = cell.migrate_x + random.gauss(0, 30)
                            ally.migrate_y = cell.migrate_y + random.gauss(0, 30)
                            ally.migrating = True
                            ally.attacks_here = 0
        else:
            cell.alert = False

            # --- Migráció mód: célzottan megy az új legelőre ---
            if cell.migrating:
                dx = cell.migrate_x - cell.x
                dy = cell.migrate_y - cell.y
                dist_to_target = math.sqrt(dx * dx + dy * dy)
                if dist_to_target < 50:
                    cell.migrating = False
                    cell.attacks_here = 0
                    # Megérkezett: van kaja a közelben? Ha nem, új migrációt indít
                    if cell.ticks_without_food > 100:
                        cell.start_migration()  # Tovább keres
                else:
                    # Útközben is figyel: ha kaját lát, megáll
                    if cell.ticks_without_food < 20:
                        cell.migrating = False  # Talált kaját útközben!
                    else:
                        cell.steer_towards(cell.migrate_x, cell.migrate_y, 0.7)
                        cell.sprinting = dist_to_target > 200 and cell.sprint_energy > 40
                        return

            # --- Legelő-megosztás + veszélyzóna megosztás csordán belül ---
            if cell.genome.social > 0.3 and allies:
                for ally, dist in allies:
                    if ally.genome.social > 0.2:
                        # Legelő megosztás
                        if ally.food_spots and not cell.food_spots:
                            cell.food_spots = [(x, y, a * 0.5) for x, y, a in ally.food_spots[:2]]
                        # Veszélyzóna megosztás
                        if ally.danger_zones and not cell.danger_zones:
                            cell.danger_zones = [(x, y, d * 0.4) for x, y, d in ally.danger_zones[:2]]
                        break

            # --- Veszélyzónából kimenekül proaktívan ---
            current_danger = cell.is_danger_zone(cell.x, cell.y)
            if current_danger > 1.5 and not cell.migrating:
                # Ezt a helyet kerülni kell — migrál
                cell.start_migration()
                return

            # Jóllakott + biztonságban = energiatakarékos mód
            energy_ratio = cell.energy / max(1, cell.genome.repro_thresh)
            if energy_ratio > 0.9 and not enemies:
                cell.thrust = min(cell.thrust, 0.15)  # Pihenős sodródás

            self._seek_food_or_wander(cell)

    def _seek_food_or_wander(self, cell):
        """Táplálék keresés vagy kóborlás. Ragadozó húst keres, növényevő növényt."""
        sense = cell.genome.sense_range
        is_pred = cell.genome.is_predator()
        # Növényevők jobban érzékelik a kaját (szaglás bónusz)
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
            # Csak a megfelelő típusú kaját keresi
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
            # Energiatakarékos kajakereső: távolság + éhség alapú sebesség
            energy_ratio = cell.energy / max(1, cell.genome.repro_thresh)
            if best_dist < cell.radius * 3:
                food_thrust = 0.2   # Nagyon közel: lassú odaúszás
            elif best_dist < 40:
                food_thrust = 0.35  # Közel: lassú
            elif energy_ratio > 0.8:
                food_thrust = 0.3   # Jóllakott: nyugisan megy
            elif energy_ratio < 0.3:
                food_thrust = 0.75  # Éhes: sietve
            else:
                food_thrust = 0.45  # Normál: közepes tempó
            cell.steer_towards(best_food[0], best_food[1], food_thrust)
            return True
        else:
            # Nincs kaja a közelben: memóriából keres legelőt
            spot = cell.best_food_spot()
            if spot and not is_pred:
                # Legelő felé: ha éhes gyorsabban, ha jóllakott lassabban
                energy_ratio = cell.energy / max(1, cell.genome.repro_thresh)
                mem_thrust = 0.3 if energy_ratio > 0.5 else 0.55
                cell.steer_towards(spot[0], spot[1], mem_thrust)
                # Ha odaért de nincs kaja, halványítja az emléket
                dx = spot[0] - cell.x
                dy = spot[1] - cell.y
                if dx * dx + dy * dy < 40 * 40:
                    cell.food_spots = [(x, y, a * 0.85) for x, y, a in cell.food_spots]
                    cell.food_spots = [(x, y, a) for x, y, a in cell.food_spots if a > 0.2]
                return True
            # Növényevő: feromon alapú kajakeresés mielőtt kóborolna
            if not is_pred:
                gx, gy = self.pheromones.read_gradient(cell.x, cell.y, PheromoneMap.FOOD_HERE)
                if abs(gx) + abs(gy) > 0.15:
                    target_x = cell.x + gx * 40
                    target_y = cell.y + gy * 40
                    cell.steer_towards(target_x, target_y, 0.4)
                    return True
            # Régóta nem evett + kóborolna → inkább migráció!
            if cell.ticks_without_food > 150 and not cell.migrating:
                cell.start_migration()
                return True
            cell.wander()
            return False

    def _eat_food(self, cell):
        """Sejt eszi a közeli táplálékot — LASSÚ evés, a kaja fokozatosan fogy."""
        eat_range = cell.radius + FOOD_RADIUS + 2
        eat_range_sq = eat_range * eat_range
        nearby_idxs = self.food_grid.query(cell.x, cell.y, eat_range + SPATIAL_GRID_SIZE)
        is_pred = cell.genome.is_predator()
        eaten_set = set()
        ate_something = False

        for idx in nearby_idxs:
            if idx >= len(self.food) or idx in eaten_set:
                continue
            food_item = self.food[idx]
            fx, fy, fe = food_item[0], food_item[1], food_item[2]
            is_meat = food_item[3] if len(food_item) > 3 else False
            if is_pred and not is_meat:
                continue
            if not is_pred and is_meat:
                continue
            dx = fx - cell.x
            dy = fy - cell.y
            if dx * dx + dy * dy < eat_range_sq:
                # Lassú evés: kiszív eat_speed energiát tick-enként
                bite = min(self.settings['eat_speed'], fe)
                # Növényevők hatékonyabban emésztik a növényeket
                digest_eff = 0.9 if not is_pred else 0.7
                cell.energy += bite * digest_eff
                new_energy = fe - bite
                ate_something = True

                if new_energy <= FOOD_MIN_ENERGY:
                    eaten_set.add(idx)
                    self.stats["total_eaten"] += 1
                else:
                    # Kaja marad, csak csökken
                    self.food[idx] = (fx, fy, new_energy, is_meat)

                cell.ticks_without_food = 0
                cell.last_eat_tick = self.tick
                # Evés közben megáll
                cell.thrust = min(cell.thrust, 0.1)
                if not is_pred:
                    cell.remember_food(fx, fy)
                    self.pheromones.deposit(fx, fy, PheromoneMap.FOOD_HERE, 0.2)
                else:
                    # Ragadozó tetemszag: más ragadozókat is odavonz
                    self.pheromones.deposit(fx, fy, PheromoneMap.FOOD_HERE, 0.3)
                break  # Egyszerre csak 1 kaját eszik

        if eaten_set:
            self.food = [f for i, f in enumerate(self.food) if i not in eaten_set]

    def _predator_attack(self, predator):
        """Ragadozó sebzi a prédát — per-tick harc, nem instant kill."""
        attack_range = predator.radius + 8
        nearby = self.cell_grid.query(predator.x, predator.y, attack_range)

        for prey in nearby:
            if prey.id == predator.id or not prey.alive:
                continue
            if prey.genome.is_predator():
                continue
            dx = prey.x - predator.x
            dy = prey.y - predator.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < predator.radius + prey.radius + 3:
                # --- Confusion effect: raj-zavar ---
                # 5+ kicsi növényevő együtt → ragadozó összezavarodik
                if prey.pack_mates >= 5:
                    small_count = 0
                    nearby_herbs = self.cell_grid.query(prey.x, prey.y, 40)
                    for h in nearby_herbs:
                        if h.alive and not h.genome.is_predator() and h.current_size < 10:
                            small_count += 1
                    if small_count >= 5 and random.random() < 0.4:
                        # Raj-zavar: elhibázza a támadást
                        predator.energy -= 0.5
                        continue

                # --- Sebzés számítás ---
                damage = predator.damage_per_tick

                # Csordavédelmi bónusz: defense aura tankoktól
                herd_defense = 0
                nearby_herbs = self.cell_grid.query(prey.x, prey.y, 50)
                for ally in nearby_herbs:
                    if ally.id != prey.id and ally.alive and not ally.genome.is_predator():
                        # Tankok (nagy + defense) védelmi aurát adnak
                        if ally.current_size > 12 and ally.genome.defense > 3:
                            herd_defense += ally.genome.defense * 0.2
                herd_defense = min(herd_defense, 4.0)

                # Prey defense csökkenti a sebzést
                defense = prey.genome.defense + herd_defense
                actual_damage = max(0.05, damage - defense * 0.15)

                # Sebzés alkalmazása
                prey.hp -= actual_damage
                prey.damaged_flash = 8
                prey.being_attacked_by.append(predator.id)
                predator.energy -= 0.3  # Támadás energiába kerül

                # --- Knockback: nagy préda meglöki a ragadozót ---
                if prey.current_size > predator.current_size * 1.3:
                    knockback = (prey.current_size - predator.current_size) * 0.15
                    if dist > 0.1:
                        predator.vx -= (dx / dist) * knockback
                        predator.vy -= (dy / dist) * knockback

                # --- Halál: préda HP elfogy → TETEM keletkezik ---
                if prey.hp <= 0:
                    prey.alive = False
                    predator.remember_hunt(prey, True, prey.x, prey.y)
                    predator.kills += 1
                    self.stats["total_died"] += 1

                    # Tetem: méret-arányos energia, amit lassan kell enni
                    # Nagy sejt = nagy tetem = sok energia = sokáig tart megenni
                    corpse_energy = prey.energy * 0.5 + prey.genome.size ** 2 * 0.8
                    corpse_energy = max(10, corpse_energy)
                    self.food.append((prey.x, prey.y, corpse_energy, True))
                break  # Egyszerre 1 prédát sebez


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

        # Kamera
        self.cam_x = 0
        self.cam_y = 0
        self.zoom = 1.0

        self.show_info = True
        self.selected_cell = None

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

        # Búvóhelyek rajzolása
        for s in world.shelters:
            sx, sy = self.world_to_screen(s['x'], s['y'])
            r = max(5, int(s['radius'] * self.zoom))
            if -r < sx < self.screen_w + r and -r < sy < self.screen_h + r:
                shelter_surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                pygame.draw.circle(shelter_surf, (50, 40, 30, 50), (r, r), r)
                pygame.draw.circle(shelter_surf, (70, 55, 35, 30), (r, r), r, max(1, int(2 * self.zoom)))
                self.screen.blit(shelter_surf, (sx - r, sy - r))

        # Feromon heatmap (halványan)
        pm = world.pheromones
        gs = pm.grid_size
        for row_i in range(pm.rows):
            for col_i in range(pm.cols):
                cell_data = pm.grid[row_i][col_i]
                if not cell_data:
                    continue
                wx = col_i * gs + gs // 2
                wy = row_i * gs + gs // 2
                sx, sy = self.world_to_screen(wx, wy)
                r = max(2, int(gs * 0.4 * self.zoom))
                if -r < sx < self.screen_w + r and -r < sy < self.screen_h + r:
                    # Veszély=piros, nyom=kék, kaja=zöld, pár=pink, préda=narancs
                    danger = cell_data.get(PheromoneMap.DANGER, 0)
                    trail = cell_data.get(PheromoneMap.TRAIL, 0)
                    food_p = cell_data.get(PheromoneMap.FOOD_HERE, 0)
                    mate_p = cell_data.get(PheromoneMap.MATE, 0)
                    prey_p = cell_data.get(PheromoneMap.PREY_SCENT, 0)
                    total = danger + trail + food_p + mate_p + prey_p
                    if total > 0.2:
                        pr = min(80, int(danger * 25 + mate_p * 20 + prey_p * 12))
                        pg = min(50, int(food_p * 20 + mate_p * 8))
                        pb = min(50, int(trail * 15 + mate_p * 15))
                        alpha = min(45, int(total * 10))
                        if alpha > 3:
                            p_surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                            pygame.draw.circle(p_surf, (pr, pg, pb, alpha), (r, r), r)
                            self.screen.blit(p_surf, (sx - r, sy - r))

        # Oázisok (legelők) halványan a háttérben
        for src in world.food_sources:
            sx, sy = self.world_to_screen(src['x'], src['y'])
            r = max(5, int(src['spread'] * 1.5 * self.zoom))
            if -r < sx < self.screen_w + r and -r < sy < self.screen_h + r:
                oasis_surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                alpha = int(15 + src['richness'] * 15)
                pygame.draw.circle(oasis_surf, (20, 60, 20, alpha), (r, r), r)
                self.screen.blit(oasis_surf, (sx - r, sy - r))

        # Táplálék rajzolása — méret tükrözi az energiát
        for food_item in world.food:
            fx, fy, fe = food_item[0], food_item[1], food_item[2]
            is_meat = food_item[3] if len(food_item) > 3 else False
            sx, sy = self.world_to_screen(fx, fy)
            if -5 < sx < self.screen_w + 5 and -5 < sy < self.screen_h + 5:
                if is_meat:
                    # Tetem: méretarányos, sötétpiros → halvány ahogy fogy
                    r = max(3, int(math.sqrt(fe) * 0.8 * self.zoom))
                    brightness = min(220, int(80 + fe * 2))
                    fade = min(255, int(100 + fe * 3))  # Elhalványul ahogy fogyasztják
                    pygame.draw.circle(self.screen, (brightness, 30, 25), (sx, sy), r)
                    # Csont körvonal a nagyobb tetemeknél
                    if r > 4:
                        pygame.draw.circle(self.screen, (fade, fade // 2, fade // 3),
                                         (sx, sy), r, max(1, int(self.zoom)))
                else:
                    # Növény: kis fűszál → nagy bokor, méretarányos
                    r = max(1, int(math.sqrt(fe) * 0.55 * self.zoom))
                    green = min(255, int(80 + fe * 4))
                    pygame.draw.circle(self.screen, (30 + min(30, int(fe)), green, 30), (sx, sy), r)

        # Falka/csorda vonalak (közeli társak között)
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
                # Közös célpont = erősebb vonal (falkavadászat)
                if cell.target_id and cell.target_id == other.target_id:
                    color = (200, 60, 60, 60)
                else:
                    color = (60, 120, 60, 35)
                link_surf = pygame.Surface((abs(sx2 - sx1) + 2, abs(sy2 - sy1) + 2), pygame.SRCALPHA)
                ox = min(sx1, sx2)
                oy = min(sy1, sy2)
                pygame.draw.line(link_surf, color, (sx1 - ox, sy1 - oy), (sx2 - ox, sy2 - oy), 1)
                self.screen.blit(link_surf, (ox, oy))

        # Egysejtűek rajzolása
        for cell in world.cells:
            if not cell.alive:
                continue
            sx, sy = self.world_to_screen(cell.x, cell.y)
            r = max(2, int(cell.radius * self.zoom))

            if -r - 20 < sx < self.screen_w + r + 20 and -r - 20 < sy < self.screen_h + r + 20:
                color = cell.genome.color()

                # Migráció jelzés (sárga nyíl) — mindkét típusnál
                if cell.migrating:
                    dx = cell.migrate_x - cell.x
                    dy = cell.migrate_y - cell.y
                    d = max(math.sqrt(dx*dx + dy*dy), 1)
                    arrow_len = min(20, d * 0.1) * self.zoom
                    ax = sx + dx / d * arrow_len
                    ay = sy + dy / d * arrow_len
                    col = (220, 200, 50) if not cell.genome.is_predator() else (255, 120, 50)
                    pygame.draw.line(self.screen, col, (sx, sy), (int(ax), int(ay)), max(1, int(self.zoom)))

                # Alfa jelzés (fehér pont a tetején)
                if cell.leader_id is None and cell.pack_mates >= 2:
                    pygame.draw.circle(self.screen, (255, 255, 255), (sx, sy - r - 3), max(1, int(2 * self.zoom)))

                # Lesbenállás jelzés (szem - sárga)
                if cell.ambushing:
                    pygame.draw.circle(self.screen, (200, 180, 50), (sx, sy), r + 2, 1)
                    # Kis szem
                    pygame.draw.circle(self.screen, (255, 220, 50), (sx + r // 2, sy - r // 3), max(1, int(self.zoom)))

                # Rejtőzés jelzés (halványabb rajz)
                if cell.hiding and not cell.genome.is_predator():
                    pygame.draw.circle(self.screen, (60, 80, 60), (sx, sy), r + 1, 1)

                # Érzékelési sugár (kiválasztott sejtnél)
                if self.selected_cell and self.selected_cell.id == cell.id:
                    sense_r = int(cell.genome.sense_range * self.zoom)
                    if sense_r > 2:
                        sense_surf = pygame.Surface((sense_r * 2, sense_r * 2), pygame.SRCALPHA)
                        pygame.draw.circle(sense_surf, (*color, 20), (sense_r, sense_r), sense_r)
                        self.screen.blit(sense_surf, (sx - sense_r, sy - sense_r))

                # --- Csillók rajzolása ---
                cilia_angles = cell.genome.cilia_positions()
                cilia_len = max(3, int((r * 0.6 + 2) * self.zoom))
                for i, ca in enumerate(cilia_angles):
                    abs_angle = cell.angle + ca
                    # Csilló hullámzás animáció
                    wave = math.sin(cell.cilia_phase + i * 1.5) * 0.3
                    draw_angle = abs_angle + wave

                    # Csilló kiindulópontja a sejt felszínén
                    cx1 = sx + int(math.cos(abs_angle) * r)
                    cy1 = sy + int(math.sin(abs_angle) * r)
                    # Csilló vége
                    cx2 = cx1 + int(math.cos(draw_angle) * cilia_len)
                    cy2 = cy1 + int(math.sin(draw_angle) * cilia_len)

                    cilia_color = tuple(min(255, c + 80) for c in color)
                    pygame.draw.line(self.screen, cilia_color, (cx1, cy1), (cx2, cy2),
                                     max(1, int(self.zoom)))

                # Sejt test
                pygame.draw.circle(self.screen, color, (sx, sy), r)

                # Belső mag (sejtmag)
                inner_r = max(1, r // 2)
                brighter = tuple(min(255, c + 50) for c in color)
                pygame.draw.circle(self.screen, brighter, (sx, sy), inner_r)

                # Irányjelző pont (hová néz)
                if r > 4:
                    dir_x = sx + int(math.cos(cell.angle) * r * 0.6)
                    dir_y = sy + int(math.sin(cell.angle) * r * 0.6)
                    pygame.draw.circle(self.screen, (255, 255, 255), (dir_x, dir_y), max(1, r // 5))

                # Ragadozó: piros szegély
                if cell.genome.is_predator():
                    pygame.draw.circle(self.screen, (255, 60, 60), (sx, sy), r, 1)

                # Sprint effekt: villogó fényes gyűrű
                if cell.sprinting:
                    sprint_color = (255, 255, 100) if cell.genome.is_predator() else (100, 255, 255)
                    pygame.draw.circle(self.screen, sprint_color, (sx, sy), r + 3, 1)

                # Sérülés villanás
                if cell.damaged_flash > 0:
                    flash_alpha = min(180, cell.damaged_flash * 25)
                    flash_surf = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
                    pygame.draw.circle(flash_surf, (255, 50, 30, flash_alpha), (r + 2, r + 2), r + 2)
                    self.screen.blit(flash_surf, (sx - r - 2, sy - r - 2))

                # Gyerek jelzés (kisebb, halványabb kontúr)
                if cell.is_child:
                    pygame.draw.circle(self.screen, (180, 180, 255), (sx, sy), r + 2, 1)

                # Párkeresés jelzés (pink szív-szerű pulzáló kör)
                if cell.seeking_mate and r > 3:
                    pulse = abs(math.sin(cell.age * 0.08)) * 0.5 + 0.5
                    mate_r = int(r + 3 + pulse * 3)
                    mate_surf = pygame.Surface((mate_r * 2, mate_r * 2), pygame.SRCALPHA)
                    pygame.draw.circle(mate_surf, (255, 100, 180, int(40 + pulse * 40)),
                                     (mate_r, mate_r), mate_r, max(1, int(2 * self.zoom)))
                    self.screen.blit(mate_surf, (sx - mate_r, sy - mate_r))

                # HP sáv (felül, piros)
                if r > 4:
                    bar_w = r * 2
                    bar_h = 2
                    hp_ratio = cell.hp_ratio
                    if hp_ratio < 1.0:  # Csak ha sérült
                        pygame.draw.rect(self.screen, (40, 40, 40),
                                         (sx - bar_w // 2, sy - r - 9, bar_w, bar_h))
                        hp_color = (200, 50, 50) if hp_ratio < 0.3 else (220, 160, 30) if hp_ratio < 0.6 else (50, 200, 50)
                        pygame.draw.rect(self.screen, hp_color,
                                         (sx - bar_w // 2, sy - r - 9, int(bar_w * hp_ratio), bar_h))

                    # Energia sáv (alatta, zöld)
                    energy_ratio = min(1, cell.energy / max(1, cell.genome.repro_thresh))
                    pygame.draw.rect(self.screen, (30, 30, 30),
                                     (sx - bar_w // 2, sy - r - 6, bar_w, bar_h))
                    e_color = (50, 180, 50) if energy_ratio > 0.3 else (180, 50, 50)
                    pygame.draw.rect(self.screen, e_color,
                                     (sx - bar_w // 2, sy - r - 6, int(bar_w * energy_ratio), bar_h))

        # Kiválasztott sejt infó
        if self.selected_cell and self.selected_cell.alive:
            self._draw_cell_info(self.selected_cell)

        # HUD
        self._draw_hud(world, paused, sim_speed)

    def _draw_hud(self, world, paused, sim_speed):
        # Felső sáv
        bar_h = 30
        bar_surf = pygame.Surface((self.screen_w, bar_h), pygame.SRCALPHA)
        bar_surf.fill((0, 0, 0, 160))
        self.screen.blit(bar_surf, (0, 0))

        # Infók
        alive = len(world.cells)
        pred = world.stats["predators"]
        herb = world.stats["herbivores"]
        food = len(world.food)
        gen = world.stats["max_generation"]

        status = "SZÜNET" if paused else f"x{sim_speed}"
        info = (f"Tick: {world.tick:,}  |  Sejtek: {alive} "
                f"(Növényevő: {herb}, Ragadozó: {pred})  |  "
                f"Táplálék: {food}  |  Gen: {gen}  |  {status}")

        text = self.font_medium.render(info, True, (200, 220, 200))
        self.screen.blit(text, (10, 6))

        # Alsó sáv - vezérlés
        bot_y = self.screen_h - 24
        bot_surf = pygame.Surface((self.screen_w, 24), pygame.SRCALPHA)
        bot_surf.fill((0, 0, 0, 140))
        self.screen.blit(bot_surf, (0, bot_y))

        controls = "SPACE: Szünet | W/S: Sebesség | Nyilak: Kamera | 1: Növényevő | 2: Ragadozó | F: Kaja | M: Beállítások | R: Reset | Scroll: Zoom | Q: Kilépés"
        ct = self.font_small.render(controls, True, (140, 140, 160))
        self.screen.blit(ct, (10, bot_y + 5))

        # Jobb oldali infó panel
        if self.show_info:
            self._draw_stats_panel(world)

    def _draw_stats_panel(self, world):
        pw, ph = 220, 200
        px = self.screen_w - pw - 10
        py = 40

        panel_surf = pygame.Surface((pw, ph), pygame.SRCALPHA)
        panel_surf.fill((0, 0, 0, 180))
        self.screen.blit(panel_surf, (px, py))
        pygame.draw.rect(self.screen, (60, 80, 100), (px, py, pw, ph), 1, border_radius=4)

        y = py + 8
        title = self.font_medium.render("Statisztikák", True, (100, 200, 255))
        self.screen.blit(title, (px + 10, y))
        y += 22

        stats_lines = [
            f"Született: {world.stats['total_born']:,}",
            f"Meghalt: {world.stats['total_died']:,}",
            f"Elfogyasztott táp: {world.stats['total_eaten']:,}",
            f"Max generáció: {world.stats['max_generation']}",
            "",
            "Átlagos genom:",
        ]

        for line in stats_lines:
            t = self.font_small.render(line, True, (180, 180, 190))
            self.screen.blit(t, (px + 10, y))
            y += 16

        # Átlagos genom értékek
        if world.cells:
            avg_size = sum(c.genome.size for c in world.cells) / len(world.cells)
            avg_spd = sum(c.genome.max_speed for c in world.cells) / len(world.cells)
            avg_sense = sum(c.genome.sense_range for c in world.cells) / len(world.cells)
            avg_cilia = sum(c.genome.num_cilia for c in world.cells) / len(world.cells)
            avg_spread = sum(c.genome.cilia_spread for c in world.cells) / len(world.cells)

            genome_lines = [
                f"  Méret: {avg_size:.1f}  MaxSpd: {avg_spd:.1f}",
                f"  Érzék: {avg_sense:.0f}  Csillók: {avg_cilia:.1f}",
                f"  Szórás: {avg_spread:.0%}",
            ]
            for line in genome_lines:
                t = self.font_small.render(line, True, (160, 180, 160))
                self.screen.blit(t, (px + 10, y))
                y += 16

    def _draw_cell_info(self, cell):
        pw, ph = 270, 400
        px = 10
        py = 40

        panel_surf = pygame.Surface((pw, ph), pygame.SRCALPHA)
        panel_surf.fill((0, 0, 0, 200))
        self.screen.blit(panel_surf, (px, py))

        color = cell.genome.color()
        pygame.draw.rect(self.screen, color, (px, py, pw, ph), 1, border_radius=4)

        y = py + 8
        kind = "Ragadozó" if cell.genome.is_predator() else "Növényevő"
        title = self.font_medium.render(f"Sejt #{cell.id} ({kind})", True, color)
        self.screen.blit(title, (px + 10, y))
        y += 20

        # Csilló típus leírás
        n = cell.genome.num_cilia
        sp = cell.genome.cilia_spread
        if sp < 0.2:
            cilia_desc = f"{n} csilló, egyirányú (torpedó)"
        elif sp < 0.5:
            cilia_desc = f"{n} csilló, szűk szórás"
        elif sp < 0.8:
            cilia_desc = f"{n} csilló, széles szórás"
        else:
            cilia_desc = f"{n} csilló, körkörös (medúza)"

        child_str = " [KÖLYÖK]" if cell.is_child else (" [PÁRT KERES]" if cell.seeking_mate else "")
        lines = [
            f"HP: {cell.hp:.0f}/{cell.max_hp:.0f}  ({cell.hp_ratio:.0%}){child_str}",
            f"Energia: {cell.energy:.0f} / {cell.genome.repro_thresh:.0f}",
            f"Kor: {cell.age}  Gen: {cell.generation}",
            f"Gyerekek: {cell.children}  Ölések: {cell.kills}",
            f"Sebzés/tick: {cell.damage_per_tick:.2f}",
            "",
            f"--- Csilló rendszer ---",
            f"  {cilia_desc}",
            f"  Erő/csilló: {cell.genome.cilia_power:.2f}",
            f"  Sebesség: {cell.genome.max_speed:.2f} x{cell.effective_speed_mult:.0%}",
            f"  Fordulás: {math.degrees(cell.genome.turn_rate):.1f}°/tick",
            f"  Manőver: {cell.genome.maneuverability:.0%}",
            "",
            f"Méret: {cell.current_size:.1f} (max:{cell.genome.size:.1f})",
            f"Érzékelés: {cell.genome.sense_range:.0f}",
            f"Támadás: {cell.genome.attack:.1f}  Védekezés: {cell.genome.defense:.1f}",
            f"Anyagcsere: {cell.genome.metabolism:.2f}",
            f"Diéta: {cell.genome.diet:.2f}",
            f"Agresszió: {cell.genome.aggression:.2f}  Társas: {cell.genome.social:.2f}",
            f"Alom: {cell.genome.litter_size}  Sprint: {cell.sprint_energy:.0f}% {'AKTÍV' if cell.sprinting else 'kész' if cell.sprint_cooldown <= 0 else f'hűtés:{cell.sprint_cooldown}'}",
            f"Társak: {cell.pack_mates}  Vezető: {'ALFA' if cell.leader_id is None and cell.pack_mates > 0 else f'#{cell.leader_id}' if cell.leader_id else '-'}  ({cell.leadership:.1f}pt)",
            f"Éhezés: {cell.ticks_without_food} tick" + (" | VÁNDOROL" if cell.migrating else "") + (" | LESBEN" if cell.ambushing else "") + (" | REJTŐZIK" if cell.hiding and not cell.ambushing else ""),
        ]
        # Ragadozó memória megjelenítés
        if cell.genome.is_predator() and (cell.prey_memory or cell.hunting_spots):
            lines.append("")
            lines.append("--- Memória ---")
            if cell.prey_memory:
                s = sum(1 for m in cell.prey_memory if m[3])
                f_count = len(cell.prey_memory) - s
                lines.append(f"  Vadászatok: {s} sikeres / {f_count} sikertelen")
            if cell.prefer_weak != 0 or cell.prefer_isolated != 0:
                prefs = []
                if cell.prefer_weak > 0.2: prefs.append("gyengét keres")
                if cell.prefer_isolated > 0.2: prefs.append("magányost keres")
                if cell.prefer_slow > 0.2: prefs.append("lassút keres")
                if prefs:
                    lines.append(f"  Tanult: {', '.join(prefs)}")
            if cell.hunting_spots:
                lines.append(f"  Vadászterületek: {len(cell.hunting_spots)}")
        # Növényevő memória
        if not cell.genome.is_predator():
            has_mem = cell.food_spots or cell.danger_zones or cell.migrating
            if has_mem:
                lines.append("")
                lines.append("--- Memória ---")
                if cell.food_spots:
                    lines.append(f"  Ismert legelők: {len(cell.food_spots)}")
                if cell.danger_zones:
                    lines.append(f"  Veszélyzónák: {len(cell.danger_zones)}")
                if cell.migrating:
                    dx = cell.migrate_x - cell.x
                    dy = cell.migrate_y - cell.y
                    md = math.sqrt(dx*dx + dy*dy)
                    lines.append(f"  MIGRÁL → ({cell.migrate_x:.0f},{cell.migrate_y:.0f}) [{md:.0f}px]")
                if cell.attacks_here > 0:
                    lines.append(f"  Támadások itt: {cell.attacks_here}")

        for line in lines:
            col = (140, 200, 255) if line.startswith("---") else (190, 190, 200)
            t = self.font_small.render(line, True, col)
            self.screen.blit(t, (px + 10, y))
            y += 15

    def handle_click(self, pos, world):
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


# --- Beállítások menü ---
class SettingsMenu:
    """Élő beállítások panel csúszkákkal."""

    ITEMS = [
        {"name": "Kaja energia",        "key": "food_energy",      "min": 5,     "max": 60,   "step": 1,     "fmt": "{:.0f}"},
        {"name": "Evési sebesség",       "key": "eat_speed",        "min": 0.2,   "max": 3.0,  "step": 0.1,   "fmt": "{:.1f}"},
        {"name": "Kaja visszanövés",     "key": "food_regrow",      "min": 0.005, "max": 0.15, "step": 0.005, "fmt": "{:.3f}"},
        {"name": "Szaporodási költség",  "key": "repro_cost",       "min": 0.15,  "max": 0.8,  "step": 0.05,  "fmt": "{:.0%}"},
        {"name": "Mutáció ráta",         "key": "mutation_rate",    "min": 0.02,  "max": 0.5,  "step": 0.02,  "fmt": "{:.0%}"},
        {"name": "Mutáció erősség",     "key": "mutation_strength", "min": 0.02,  "max": 0.4,  "step": 0.02,  "fmt": "{:.0%}"},
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
        # Háttér
        overlay = pygame.Surface((self.screen_w, self.screen_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        screen.blit(overlay, (0, 0))

        px, py = self.panel_x, self.panel_y
        pw, ph = self.panel_w, self.panel_h

        # Panel háttér
        panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
        panel.fill((20, 25, 35, 240))
        screen.blit(panel, (px, py))
        pygame.draw.rect(screen, (80, 120, 180), (px, py, pw, ph), 2, border_radius=6)

        # Cím
        title = self.font_title.render("Beállítások  (M: bezár)", True, (100, 200, 255))
        screen.blit(title, (px + 15, py + 10))

        # Sorok
        for i, item in enumerate(self.ITEMS):
            row_y = py + self.header_h + i * self.row_h
            val = settings.get(item['key'], 0)

            # Hover kiemelés
            if i == self.hovered_item:
                highlight = pygame.Surface((pw - 4, self.row_h - 2), pygame.SRCALPHA)
                highlight.fill((50, 70, 100, 80))
                screen.blit(highlight, (px + 2, row_y))

            # Név
            name_surf = self.font.render(item['name'], True, (190, 195, 210))
            screen.blit(name_surf, (px + 15, row_y + 4))

            # Érték
            val_str = item['fmt'].format(val)
            val_surf = self.font.render(val_str, True, (255, 240, 150))
            screen.blit(val_surf, (px + 195, row_y + 4))

            # [-] gomb
            btn_y = row_y + 3
            btn_w, btn_h = 28, 22
            minus_x = px + pw - 80
            plus_x = px + pw - 42

            # Csúszka háttér sáv
            bar_x = px + 240
            bar_w = pw - 240 - 10
            bar_y = row_y + 22
            pygame.draw.rect(screen, (40, 50, 65), (bar_x, bar_y, bar_w, 6), border_radius=3)
            # Csúszka pozíció
            ratio = (val - item['min']) / max(0.001, item['max'] - item['min'])
            fill_w = int(bar_w * ratio)
            color = (80, 200, 120) if ratio < 0.5 else (200, 200, 60) if ratio < 0.8 else (220, 100, 80)
            pygame.draw.rect(screen, color, (bar_x, bar_y, fill_w, 6), border_radius=3)

            # [-] és [+] gombok
            pygame.draw.rect(screen, (60, 70, 90), (minus_x, btn_y, btn_w, btn_h), border_radius=3)
            pygame.draw.rect(screen, (60, 70, 90), (plus_x, btn_y, btn_w, btn_h), border_radius=3)
            pygame.draw.rect(screen, (100, 120, 150), (minus_x, btn_y, btn_w, btn_h), 1, border_radius=3)
            pygame.draw.rect(screen, (100, 120, 150), (plus_x, btn_y, btn_w, btn_h), 1, border_radius=3)

            minus_t = self.font.render(" -", True, (200, 100, 100))
            plus_t = self.font.render(" +", True, (100, 200, 100))
            screen.blit(minus_t, (minus_x + 4, btn_y + 2))
            screen.blit(plus_t, (plus_x + 4, btn_y + 2))

    def handle_click(self, pos, settings):
        """Kattintás kezelés. Visszatér True ha valamit módosított."""
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

            # [-] gomb
            if minus_x <= mx <= minus_x + btn_w and btn_y <= my <= btn_y + btn_h:
                val = settings[item['key']]
                settings[item['key']] = max(item['min'], val - item['step'])
                return True
            # [+] gomb
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
        """Menün belül kattintott-e."""
        mx, my = pos
        return (self.panel_x <= mx <= self.panel_x + self.panel_w and
                self.panel_y <= my <= self.panel_y + self.panel_h)


# --- Game ---
class Game:
    def __init__(self):
        pygame.init()
        info = pygame.display.Info()
        # Fullscreen mód
        self.screen_w = info.current_w
        self.screen_h = info.current_h
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h), pygame.FULLSCREEN | pygame.SCALED)
        pygame.display.set_caption("Egysejtű Evolúció - Ősleves Ökoszisztéma")
        self.clock = pygame.time.Clock()

        # Világ nagyobb mint a képernyő
        self.world_w = self.screen_w * 2
        self.world_h = self.screen_h * 2

        self.world = World(self.world_w, self.world_h)
        self.renderer = Renderer(self.screen, self.world_w, self.world_h)
        self.settings_menu = SettingsMenu(self.screen_w, self.screen_h)

        self.paused = False
        self.sim_speed = 1
        self.running = True

        # Kamera középre
        self.renderer.cam_x = (self.world_w - self.screen_w) / 2
        self.renderer.cam_y = (self.world_h - self.screen_h) / 2

    def run(self):
        while self.running:
            self._handle_events()
            if not self.paused:
                for _ in range(self.sim_speed):
                    self.world.update()
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
                    self.sim_speed = max(1, self.sim_speed - 1)
                elif event.key == pygame.K_f:
                    for _ in range(3):
                        self.world._spawn_food_cluster()
                elif event.key == pygame.K_1:
                    # Növényevő spawn a képernyő közepén
                    wx, wy = self.renderer.screen_to_world(self.screen_w // 2, self.screen_h // 2)
                    for _ in range(5):
                        self.world.spawn_herbivore(
                            wx + random.gauss(0, 30),
                            wy + random.gauss(0, 30))
                elif event.key == pygame.K_2:
                    # Ragadozó spawn a képernyő közepén
                    wx, wy = self.renderer.screen_to_world(self.screen_w // 2, self.screen_h // 2)
                    for _ in range(3):
                        self.world.spawn_predator(
                            wx + random.gauss(0, 30),
                            wy + random.gauss(0, 30))
                elif event.key == pygame.K_r:
                    self.world = World(self.world_w, self.world_h)
                    self.renderer.selected_cell = None
                elif event.key == pygame.K_TAB:
                    self.renderer.show_info = not self.renderer.show_info
                elif event.key == pygame.K_m:
                    self.settings_menu.toggle()
                    if self.settings_menu.visible:
                        self.paused = True  # Menü megnyitásakor szünet
            elif event.type == pygame.MOUSEMOTION:
                self.settings_menu.handle_mouse_move(event.pos)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # Ha a menü nyitva van és belekattintottunk
                    if self.settings_menu.visible:
                        if self.settings_menu.is_click_inside(event.pos):
                            self.settings_menu.handle_click(event.pos, self.world.settings)
                            # Globális konstansok szinkronizálása
                            global MUTATION_RATE, MUTATION_STRENGTH
                            MUTATION_RATE = self.world.settings['mutation_rate']
                            MUTATION_STRENGTH = self.world.settings['mutation_strength']
                            continue
                        else:
                            self.settings_menu.visible = False  # Kívül kattintás: bezár
                            continue
                    self.renderer.handle_click(event.pos, self.world)
                elif event.button == 4:  # Scroll up - zoom in
                    self.renderer.zoom = min(3.0, self.renderer.zoom * 1.1)
                elif event.button == 5:  # Scroll down - zoom out
                    self.renderer.zoom = max(0.3, self.renderer.zoom / 1.1)

        # Kamera mozgatás nyilakkal + egérrel a széleken
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

        # Egérrel is mozgathat a széleken
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

        # Kamera határok
        self.renderer.cam_x = max(0, min(self.world_w - self.screen_w / self.renderer.zoom,
                                          self.renderer.cam_x))
        self.renderer.cam_y = max(0, min(self.world_h - self.screen_h / self.renderer.zoom,
                                          self.renderer.cam_y))


def main():
    game = Game()
    game.run()


if __name__ == "__main__":
    main()
