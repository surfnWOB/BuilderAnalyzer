# BuilderAnalyzer

Analyzes Team Builders from Pokemon Showdown, outputs sets compendium, builder statistics, sorted builders, and archetype analysis.

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) - Python package manager

## Installation

1. Install uv (if not already installed):

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. Get the code:

**Option A: Clone with git**
```bash
git clone https://github.com/surfnWOB/BuilderAnalyzer.git
cd BuilderAnalyzer
```

**Option B: Download as ZIP**
- Click the green "Code" button on GitHub and select "Download ZIP"
- Extract the ZIP file and open a terminal in that folder

3. Install dependencies:

```bash
uv sync
```

## Usage

1. Place your Pokemon Showdown builder export in `my_builder.txt` (or specify a different file with `-i`)

2. Run the analyzer:

```bash
uv run python main.py
```

### Command Line Options

```
--input, -i FILE    Input builder file (default: my_builder.txt)
--no-download       Skip downloading Pokemon data files
--force-download    Force re-download of Pokemon data files even if fresh
--no-archetypes     Skip archetype analysis
--verbose, -v       Enable verbose logging
```

### Examples

```bash
# Analyze default builder
uv run python main.py

# Analyze a specific file
uv run python main.py -i my_other_builder.txt

# Skip archetype analysis for faster processing
uv run python main.py --no-archetypes

# Force refresh of Pokemon data
uv run python main.py --force-download
```

## Output

Results are written to the `output/` directory, organized by generation:

```
output/
  gen3ou/
    statistics/     # Usage statistics (txt and csv)
    sets/           # Sets compendium files
    builders/       # Sorted builder output
    archetypes/     # Archetype analysis results
    plots/          # FPC plots for archetype analysis
```

## Configuration

Advanced configuration can be done by editing `src/config.py`. Key options include:

### Team Completeness
- `anomaly_threshold`: How strict to be about incomplete teams (0 = strictest)
- `include_incomplete_teams`: Whether to include incomplete teams in output

### Sets Compendium
- `ev_threshold`: EV difference threshold for combining similar sets
- `combine_moves`: Number of move slots that can differ when combining sets
- `ignore_sets_fraction`: Filter out least-used sets

### Archetype Analysis
- `auto_num_archetypes`: Automatically determine optimal number of archetypes
- `num_archetypes`: Manual override for number of archetypes
- `exponent`: Fuzziness parameter for clustering
- `gamma_spectral`: Weight for frequent pairs in spectral decomposition

### Builder Sorting
- `sort_builder`: Enable/disable builder sorting
- `sort_gen_by_frequency`: Sort generations by team count
- `sort_teams_by_core`: Sort teams by core synergy scores

## Data Files

Pokemon data (pokedex, items, moves, abilities) is automatically downloaded from Pokemon Showdown on first run. Files are cached in the `data/` directory and refreshed if older than 7 days.

## Credits

Original author: vapicuno
