"""
Configuration parameters for BuilderAnalyzer.
All tunable parameters are defined here for easy modification.
"""

import logging
from typing import Dict, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration class holding all parameters for the analyzer."""

    def __init__(self) -> None:
        # --- INPUT FILE ---
        self.fin: str = 'my_builder.txt'

        # --- DOWNLOAD SETTINGS ---
        self.download_pokedex: bool = True

        # --- METAGAME PARAMETERS ---
        self.all_generations: bool = True
        self.generation: List[str] = ['gen3uubl']

        # --- UNFINISHED TEAMS ---
        self.anomaly_threshold: int = 0
        self.include_incomplete_teams: bool = True

        # --- SETS PARAMETERS ---
        # Set combining
        self.ev_threshold: int = 40
        self.iv_threshold: int = 999
        self.combine_moves: int = 2

        # Move sort
        self.sort_moves_by_frequency: int = -1
        self.sort_moves_by_alphabetical: int = 0

        # Display
        self.show_shiny: bool = True
        self.show_ivs: bool = False
        self.show_nicknames: bool = False  # Don't show nicknames in output
        self.ignore_sets_fraction: List[float] = [1/8, 1/16, 1/32, 0]
        self.show_statistics_in_sets: bool = True
        self.print_archetype_label: bool = False

        # --- CORE STATISTICS PARAMETERS ---
        self.max_core_num: int = 4
        self.usage_weight: List[float] = [1, 1.5, 1.5, 1.5, 2, 2]
        self.important_items: List[str] = ['Choice Band', 'Choice Scarf', 'Choice Specs', 'Assault Vest', 'Rocky Helmet', 'Z']
        self.nature_ev_modifier: int = 120
        self.move_pair_synergy_threshold: float = 1/6
        self.move_pair_in_triplet_synergy_threshold: float = 1/6
        self.move_triplet_synergy_threshold: float = 1/6
        self.move_prob_threshold: float = 0.2
        self.move_prob_in_triplet_threshold: float = 0.1
        self.move_count_threshold: int = 2
        self.sum_move_prob_threshold: float = 0.8
        self.sum_move_prob_triplet_threshold: float = 0.8
        self.naming_ignore_category_num: int = 2
        self.naming_exclusion_move_threshold: float = 1/4 * 0.25
        self.naming_min_move_prob: float = 1/4 * 0.8
        self.naming_exclusion_cat_threshold: float = 0.1
        self.show_missing_mon_cores: bool = False
        self.show_missing_set_cores: bool = False
        self.max_missing_mon_cores: int = 2
        self.max_missing_set_cores: int = 2

        # --- ARCHETYPE STATISTICS PARAMETERS ---
        self.analyze_archetypes: bool = True
        self.exponent: float = 1.6
        self.gamma_spectral: int = 2
        self.num_archetypes_range: int = 15
        self.num_archetypes: int = 8
        self.auto_num_archetypes: bool = True  # Automatically determine optimal number
        self.min_fpc_threshold: float = 0.5  # Minimum FPC for auto selection
        self.gamma_archetypes: float = 0.7

        # --- COMBINED BUILDER PARAMETERS ---
        self.sort_builder: bool = True

        # Metagame sorting
        self.sort_gen_by_frequency: int = -1
        self.sort_gen_by_alphabetical: int = 0

        # Folder sorting
        self.sort_folder_by_frequency: int = 0
        self.sort_folder_by_alphabetical: int = 0

        # Team sorting
        self.sort_teams_by_archetype: int = 0
        self.gamma_team_assignment: int = 2
        self.metric_archetypes: int = 1
        self.sort_teams_by_lead_frequency_team_preview: int = 0
        self.sort_teams_by_lead_frequency_no_team_preview: int = -1
        self.sort_teams_by_core: int = -1
        self.sort_teams_by_alphabetical: int = 0
        self.core_number: int = 2

        # Pokemon sorting within teams
        self.sort_mons_by_frequency: int = -1
        self.sort_mons_by_color: bool = False
        self.gamma: int = 1

        logger.info("Configuration initialized with default values")

    def get_output_template(self) -> str:
        """Get the output file template name based on input file."""
        return self.fin[:self.fin.rfind('.')]

    def log_config(self) -> None:
        """Log the current configuration."""
        logger.info("Current configuration:")
        logger.info(f"  Input file: {self.fin}")
        logger.info(f"  Download pokedex: {self.download_pokedex}")
        logger.info(f"  All generations: {self.all_generations}")
        logger.info(f"  Analyze archetypes: {self.analyze_archetypes}")
        logger.info(f"  Sort builder: {self.sort_builder}")


# Nature modifiers for EV calculations
NATURE_DICT: Dict[str, List[int]] = {
    'Hardy': [0, 0, 0, 0, 0, 0],
    'Lonely': [0, 1, -1, 0, 0, 0],
    'Brave': [0, 1, 0, 0, 0, -1],
    'Adamant': [0, 1, 0, -1, 0, 0],
    'Naughty': [0, 1, 0, 0, -1, 0],
    'Bold': [0, -1, 1, 0, 0, 0],
    'Docile': [0, 0, 0, 0, 0, 0],
    'Relaxed': [0, 0, 1, 0, 0, -1],
    'Impish': [0, 0, 1, -1, 0, 0],
    'Lax': [0, 0, 1, 0, -1, 0],
    'Timid': [0, -1, 0, 0, 0, 1],
    'Hasty': [0, 0, -1, 0, 0, 1],
    'Serious': [0, 0, 0, 0, 0, 0],
    'Jolly': [0, 0, 0, -1, 0, 1],
    'Naive': [0, 0, 0, 0, -1, 1],
    'Modest': [0, -1, 0, 1, 0, 0],
    'Mild': [0, 0, -1, 1, 0, 0],
    'Quiet': [0, 0, 0, 1, 0, -1],
    'Bashful': [0, 0, 0, 0, 0, 0],
    'Rash': [0, 0, 0, 1, -1, 0],
    'Calm': [0, -1, 0, 0, 1, 0],
    'Gentle': [0, 0, -1, 0, 1, 0],
    'Sassy': [0, 0, 0, 0, 1, -1],
    'Careful': [0, 0, 0, -1, 1, 0],
    'Quirky': [0, 0, 0, 0, 0, 0]
}

# Stat index mappings
STAT_TO_INDEX: Dict[str, int] = {
    'HP': 0,
    'Atk': 1,
    'Def': 2,
    'SpA': 3,
    'SpD': 4,
    'Spe': 5
}

INDEX_TO_STAT: Dict[int, str] = {v: k for k, v in STAT_TO_INDEX.items()}
