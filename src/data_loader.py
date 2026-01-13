"""
Data loading module for BuilderAnalyzer.
Handles downloading and loading Pokemon Showdown data files.
"""

import os
import time
import urllib.request
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# URLs for Pokemon Showdown data
POKEDEX_URL = 'https://raw.githubusercontent.com/smogon/pokemon-showdown/master/data/pokedex.ts'
ITEMS_URL = 'https://raw.githubusercontent.com/smogon/pokemon-showdown/master/data/items.ts'
MOVES_URL = 'https://raw.githubusercontent.com/smogon/pokemon-showdown/master/data/moves.ts'
ABILITIES_URL = 'https://raw.githubusercontent.com/smogon/pokemon-showdown/master/data/abilities.ts'

# Data directory
DATA_DIR = 'data'

# Default max age in days before redownloading
DEFAULT_MAX_AGE_DAYS = 7


class DataLoader:
    """Handles loading Pokemon data from files and optional downloading."""

    def __init__(self) -> None:
        self.pokedex_str: Optional[str] = None
        self.items_str: Optional[str] = None
        self.moves_str: Optional[str] = None
        self.abilities_str: Optional[str] = None
        self.colors: Optional[Dict[str, Any]] = None

    def data_needs_update(self, max_age_days: int = DEFAULT_MAX_AGE_DAYS) -> bool:
        """
        Check if data files need to be downloaded or updated.

        Args:
            max_age_days: Maximum age in days before data is considered stale

        Returns:
            True if data needs to be downloaded/updated, False otherwise
        """
        required_files = ['pokedex.ts', 'items.ts', 'moves.ts', 'abilities.ts']

        for filename in required_files:
            filepath = os.path.join(DATA_DIR, filename)
            if not os.path.exists(filepath):
                logger.info(f"Data file {filename} not found, download needed")
                return True

            # Check file age
            file_age_seconds = time.time() - os.path.getmtime(filepath)
            file_age_days = file_age_seconds / (24 * 60 * 60)

            if file_age_days > max_age_days:
                logger.info(f"Data file {filename} is {file_age_days:.1f} days old, update needed")
                return True

        logger.info(f"Data files are up to date (less than {max_age_days} days old)")
        return False

    def download_data(self, force: bool = False, max_age_days: int = DEFAULT_MAX_AGE_DAYS) -> None:
        """
        Download Pokemon data files if needed.

        Args:
            force: If True, always download regardless of file age
            max_age_days: Maximum age in days before data is considered stale
        """
        if not force and not self.data_needs_update(max_age_days):
            logger.info("Skipping download - data files are fresh")
            return

        os.makedirs(DATA_DIR, exist_ok=True)

        logger.info("Beginning pokedex download...")
        try:
            urllib.request.urlretrieve(POKEDEX_URL, os.path.join(DATA_DIR, 'pokedex.ts'))
            logger.info("Pokedex downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download pokedex: {e}")
            raise

        logger.info("Beginning items download...")
        try:
            urllib.request.urlretrieve(ITEMS_URL, os.path.join(DATA_DIR, 'items.ts'))
            logger.info("Items downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download items: {e}")
            raise

        logger.info("Beginning moves download...")
        try:
            urllib.request.urlretrieve(MOVES_URL, os.path.join(DATA_DIR, 'moves.ts'))
            logger.info("Moves downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download moves: {e}")
            raise

        logger.info("Beginning abilities download...")
        try:
            urllib.request.urlretrieve(ABILITIES_URL, os.path.join(DATA_DIR, 'abilities.ts'))
            logger.info("Abilities downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download abilities: {e}")
            raise

        logger.info("All data files downloaded successfully")

    def load_data(self, load_colors: bool = False) -> None:
        """Load all Pokemon data files into memory."""
        logger.info("Loading pokedex data...")
        try:
            with open(os.path.join(DATA_DIR, 'pokedex.ts'), encoding='utf-8', errors='ignore') as f:
                self.pokedex_str = f.read()
            logger.info(f"Pokedex loaded: {len(self.pokedex_str)} characters")
        except FileNotFoundError:
            logger.error("pokedex.ts not found. Run with download_pokedex=True first.")
            raise

        logger.info("Loading items data...")
        try:
            with open(os.path.join(DATA_DIR, 'items.ts'), encoding='utf-8', errors='ignore') as f:
                self.items_str = f.read()
            logger.info(f"Items loaded: {len(self.items_str)} characters")
        except FileNotFoundError:
            logger.error("items.ts not found. Run with download_pokedex=True first.")
            raise

        logger.info("Loading moves data...")
        try:
            with open(os.path.join(DATA_DIR, 'moves.ts'), encoding='utf-8', errors='ignore') as f:
                self.moves_str = f.read()
            logger.info(f"Moves loaded: {len(self.moves_str)} characters")
        except FileNotFoundError:
            logger.error("moves.ts not found. Run with download_pokedex=True first.")
            raise

        logger.info("Loading abilities data...")
        try:
            with open(os.path.join(DATA_DIR, 'abilities.ts'), encoding='utf-8', errors='ignore') as f:
                self.abilities_str = f.read()
            logger.info(f"Abilities loaded: {len(self.abilities_str)} characters")
        except FileNotFoundError:
            logger.error("abilities.ts not found. Run with download_pokedex=True first.")
            raise

        if load_colors:
            logger.info("Loading colors data...")
            try:
                with open(os.path.join(DATA_DIR, 'colors.js'), encoding='utf-8', errors='ignore') as f:
                    colors_str = f.read()
                self.colors = json.loads(colors_str)
                logger.info(f"Colors loaded: {len(self.colors)} entries")
            except FileNotFoundError:
                logger.warning("colors.js not found. Color sorting will be disabled.")
                self.colors = None

        logger.info("All data files loaded successfully")

    def get_data(self) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        """Return all loaded data as a tuple."""
        return (self.pokedex_str, self.items_str, self.moves_str,
                self.abilities_str, self.colors)


def detect_input_format(filename: str) -> bool:
    """
    Detect whether the input file is in dense or readable format.

    Args:
        filename: Path to the builder file

    Returns:
        bool: True if dense format, False if readable format
    """
    logger.info(f"Detecting input format for {filename}...")

    with open(filename, encoding='utf-8', errors='ignore') as f:
        line = f.readline()
        while line:
            if line[0:3] == '===':
                logger.info("Detected readable format (=== markers)")
                return False
            if line[0:3] == 'gen':
                logger.info("Detected dense format (gen prefix)")
                return True
            line = f.readline()

    logger.warning("Could not detect format, defaulting to readable")
    return False


def get_generations_from_file(filename: str, input_format_dense: bool) -> List[str]:
    """
    Extract all generation identifiers from the builder file.

    Args:
        filename: Path to the builder file
        input_format_dense: Whether the file is in dense format

    Returns:
        list: List of generation strings (e.g., ['gen3ou', 'gen4ou'])
    """
    logger.info("Extracting generations from file...")
    generations: List[str] = []

    with open(filename, encoding='utf-8', errors='ignore') as f:
        line = f.readline()
        if input_format_dense:
            while line:
                if line.find(']') > -1:
                    g = line[0:line.find(']')]
                    # Skip box generations (e.g., "gen3uubl-box")
                    if g not in generations and not g.endswith('-box'):
                        generations.append(g)
                        logger.debug(f"Found generation: {g}")
                    elif g.endswith('-box'):
                        logger.debug(f"Skipping box generation: {g}")
                line = f.readline()
        else:
            while line:
                if line.find('=== [') > -1:
                    if line[0:5] == '=== [' and line[-4:-1] == '===':
                        g = line[5:line.find(']')]
                        # Skip box generations (e.g., "gen3uubl-box")
                        if g not in generations and not g.endswith('-box'):
                            generations.append(g)
                            logger.debug(f"Found generation: {g}")
                        elif g.endswith('-box'):
                            logger.debug(f"Skipping box generation: {g}")
                line = f.readline()

    logger.info(f"Found {len(generations)} generations: {generations}")
    return generations
