# BuilderAnalyzer

A Python tool for analyzing Pokemon team builders from Pokemon Showdown.

## Project Overview

BuilderAnalyzer takes user-created team builder files (in Pokemon Showdown format) and performs comprehensive analysis, producing:
- **Sets compendium** - Unique Pokemon sets with usage statistics
- **Builder statistics** - Core composition analysis (pairs, triplets, quads)
- **Sorted builders** - Reorganized team data by various metrics
- **Archetype analysis** - Team clustering using spectral fuzzy clustering

## Quick Start

```bash
# Install dependencies
uv sync

# Run with default settings (downloads Pokemon data if needed)
uv run python main.py

# Run without downloading (if data files exist and are fresh)
uv run python main.py --no-download

# Force re-download of Pokemon data
uv run python main.py --force-download

# Run without archetype analysis (faster)
uv run python main.py --no-archetypes

# Specify input file
uv run python main.py --input my_teams.txt

# Verbose logging
uv run python main.py -v
```

## Project Structure

```
BuilderAnalyzer/
├── main.py              # Entry point - orchestrates the analysis
├── pyproject.toml       # Project dependencies and metadata
├── my_builder.txt       # Example input file
├── src/                 # Source modules
│   ├── __init__.py
│   ├── config.py        # Configuration parameters and constants
│   ├── data_loader.py   # Downloads and loads Pokemon Showdown data
│   ├── set_parser.py    # Parses Pokemon sets from text
│   ├── team_analyzer.py # Team extraction and completeness analysis
│   ├── statistics.py    # MPMI calculations, set categorization
│   ├── archetype.py     # Spectral clustering for archetypes
│   └── output.py        # File output functions
├── data/                # Pokemon data files (auto-downloaded)
│   ├── pokedex.ts
│   ├── items.ts
│   ├── moves.ts
│   └── abilities.ts
└── output/              # Generated output (organized by generation)
    └── {gen}/
        ├── statistics/  # Usage and synergy statistics
        ├── sets/        # Sets compendium files
        ├── builders/    # Sorted builder output
        ├── archetypes/  # Archetype analysis results
        └── plots/       # FPC plots for archetype analysis
```

## Module Descriptions

### src/config.py
Contains the `Config` class with all tunable parameters:
- Input/output settings
- Team completeness thresholds
- Set similarity metrics
- Core statistics limits
- Archetype analysis parameters (including auto-selection)
- Sorting options

### src/data_loader.py
- Downloads Pokemon data from Pokemon Showdown GitHub
- Smart caching: skips download if files are fresh (< 7 days old)
- Loads pokedex, items, moves, abilities data
- Detects input format (dense vs readable)
- Extracts generation list from builder file
- Filters out box generations (e.g., `gen3uubl-box`)

### src/set_parser.py
- `SetParser` class for parsing Pokemon sets
- Supports dense pipe-delimited format
- Supports readable newline format
- Uses tab-prefixed move lookup for accurate matching
- `print_set()` function for output formatting

### src/team_analyzer.py
- `TeamExtractor` class for extracting teams from files
- `check_team_completeness()` - identifies incomplete teams
- `find_cores_and_leads()` - calculates team compositions

### src/statistics.py
- `calculate_mpmi()` - Multivariate Pointwise Mutual Information
- `categorize_sets()` - Groups sets by item/EVs
- `aggregate_sets_by_ev()` - Combines similar EV spreads
- `aggregate_sets_by_moves()` - Combines similar movesets

### src/archetype.py
- `ArchetypeAnalyzer` class for spectral clustering
- Uses numpy, scikit-fuzzy, matplotlib
- Auto-selects optimal number of archetypes using FPC knee detection
- Non-blocking plot generation (saves to file)
- `find_team_archetype()` - assigns teams to archetypes

### src/output.py
- `OutputWriter` class for all file outputs
- Statistics (TXT and CSV formats)
- Sets compendium files
- Sorted builder files
- Archetype statistics

## Input Formats

### Dense Format (default)
```
gen3ou]Team Name|Pokemon1Data]Pokemon2Data]...
```

### Readable Format
```
=== [gen3ou] Team Name ===

Pokemon @ Item
Ability: ...
EVs: ...
Nature Nature
- Move 1
- Move 2
...
```

## Key Configuration Options

Edit `src/config.py` or pass command-line arguments:

```python
# In src/config.py
config.download_pokedex = True      # Download fresh data
config.all_generations = True       # Process all gens in file
config.analyze_archetypes = True    # Enable clustering
config.auto_num_archetypes = True   # Auto-select cluster count
config.anomaly_threshold = 0        # 0 = strict, 999 = include all
```

## Dependencies

- Python >= 3.13
- numpy >= 2.4.1
- scikit-fuzzy >= 0.5.0
- scipy >= 1.17.0
- matplotlib >= 3.10.8
- rich >= 13.0.0

Install with: `uv sync`

## Output Files

For each generation in `output/{gen}/`:
- `statistics/usage_statistics.txt` - Pokemon usage stats
- `statistics/usage_statistics.csv` - Pokemon usage stats (CSV)
- `statistics/synergy_statistics.txt` - Core synergy stats
- `statistics/synergy_statistics.csv` - Core synergy stats (CSV)
- `sets/sets.txt` - Full sets compendium
- `sets/sets_cut_*.txt` - Filtered sets compendium
- `builders/sorted_builder.txt` - Reorganized teams
- `archetypes/archetype_statistics.txt` - Cluster analysis
- `plots/fpc_plot.png` - Clustering quality plot

## Troubleshooting

### Code hangs during archetype analysis
The old code used `plt.show()` which blocks. The new code uses `matplotlib.use('Agg')` and saves plots to files instead.

### Missing Pokemon data files
Data is auto-downloaded on first run and cached for 7 days. Use `--force-download` to refresh.

### Memory issues with large builders
- Disable archetype analysis: `--no-archetypes`
- Process specific generations by editing `config.generation`

## Development Notes

- Original author: vapicuno
- Refactored for modularity and logging (2024)
- Uses dataclasses for cleaner data structures
- Comprehensive logging to stdout (with rich formatting) and `builder_analyzer.log`
- Type hints throughout for better IDE support
