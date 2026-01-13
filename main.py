#!/usr/bin/env python3
"""
BuilderAnalyzer - Pokemon Team Builder Analysis Tool
====================================================

Analyzes Team Builders from Pokemon Showdown, outputs:
- Sets compendium
- Builder statistics
- Sorted builders
- Archetype analysis (optional)

Usage:
    python main.py [--no-download] [--no-archetypes] [--input FILE]

Author: vapicuno (original), refactored 2024
"""

import argparse
import copy
import math
import logging
import os
import sys
from itertools import combinations
from typing import List, Dict, Tuple, Any

from src.config import Config, NATURE_DICT, INDEX_TO_STAT
from src.data_loader import DataLoader, detect_input_format, get_generations_from_file
from src.set_parser import SetParser, print_set
from src.team_analyzer import TeamExtractor, check_team_completeness, find_cores_and_leads
from src.statistics import (
    calculate_mpmi, categorize_sets, aggregate_sets_by_ev,
    aggregate_sets_by_moves, sort_sets_by_name_and_count, count_move_frequency
)
from src.archetype import ArchetypeAnalyzer, ARCHETYPE_AVAILABLE
from src.output import OutputWriter

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table

# Setup rich console
console = Console()

# Setup logging with rich
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    datefmt='[%X]',
    handlers=[
        RichHandler(console=console, rich_tracebacks=True, show_path=False),
        logging.FileHandler('builder_analyzer.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze Pokemon Showdown team builders'
    )
    parser.add_argument(
        '--no-download', action='store_true',
        help='Skip downloading Pokemon data files'
    )
    parser.add_argument(
        '--force-download', action='store_true',
        help='Force re-download of Pokemon data files even if fresh'
    )
    parser.add_argument(
        '--no-archetypes', action='store_true',
        help='Skip archetype analysis'
    )
    parser.add_argument(
        '--input', '-i', type=str, default=None,
        help='Input builder file (default: my_builder.txt)'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose logging'
    )
    return parser.parse_args()


def process_generation(gen: str, config: Config, data_loader: DataLoader,
                       set_parser: SetParser, output_writer: OutputWriter,
                       is_dense_format: bool, all_incomplete_sets: List[dict],
                       all_incomplete_teams: List[dict]) -> int:
    """
    Process a single generation.

    Args:
        gen: Generation string (e.g., 'gen3ou')
        config: Configuration object
        data_loader: Data loader instance
        set_parser: Set parser instance
        output_writer: Output writer instance
        is_dense_format: Whether the input is in dense format
        all_incomplete_sets: List to append incomplete sets to
        all_incomplete_teams: List to append incomplete teams to

    Returns:
        Number of teams processed
    """
    logger.info(f"{'='*60}")
    logger.info(f"Processing generation: {gen}")
    logger.info(f"{'='*60}")

    team_preview = int(gen[3]) >= 5 if gen[3].isdigit() else True

    # Extract teams and sets
    extractor = TeamExtractor(set_parser)
    set_list, team_list, folder_count = extractor.extract_teams(
        config.fin, gen, is_dense_format
    )

    if not team_list:
        logger.warning(f"No teams found for {gen}")
        return 0

    # Check team completeness
    incomplete_sets, incomplete_teams = check_team_completeness(
        team_list, set_list, gen, config.anomaly_threshold
    )
    all_incomplete_sets.extend(incomplete_sets)
    all_incomplete_teams.extend(incomplete_teams)

    # Find cores and leads
    core_list, lead_list = find_cores_and_leads(
        team_list, set_list, config.anomaly_threshold, team_preview,
        config.show_missing_mon_cores, config.max_missing_mon_cores
    )

    # Calculate core counts and MPMI
    core_count = [sum(core_list[p].values()) for p in range(6)]
    multiplicity = [math.factorial(p + 1) for p in range(6)]
    mpmi_list = calculate_mpmi(core_list, core_count, multiplicity, config.show_missing_mon_cores)

    # Extract completed sets for analysis
    set_list_complete = []
    for team in team_list:
        if team['Anomalies'] <= config.anomaly_threshold:
            set_list_complete.extend(set_list[team['Index'][0]:team['Index'][1]])

    logger.info(f"Complete sets for analysis: {len(set_list_complete)}")

    # Sort sets by name
    set_list_sorted = sorted(copy.deepcopy(set_list_complete), key=lambda x: x['Name'])

    # Categorize sets
    category_dict = categorize_sets(set_list_sorted, config)

    # Generate category nicknames
    category_nics = generate_category_nicknames(category_dict, config)

    # Aggregate sets
    set_list_ev = aggregate_sets_by_ev(set_list_sorted, config)
    set_list_ev_sorted = sort_sets_by_name_and_count(set_list_ev)

    # Count move frequency
    move_frequency, mon_frequency = count_move_frequency(set_list_ev_sorted)

    # Aggregate by moves
    set_list_moves1 = aggregate_sets_by_moves(set_list_ev_sorted, config, slots=1)
    set_list_moves1_sorted = sort_sets_by_name_and_count(set_list_moves1)
    set_list_moves2 = aggregate_sets_by_moves(set_list_moves1_sorted, config, slots=2)
    set_list_moves2_sorted = sort_sets_by_name_and_count(set_list_moves2)

    # Archetype analysis
    archetype_analyzer = None
    if config.analyze_archetypes and ARCHETYPE_AVAILABLE:
        # Build category core list for archetype analysis
        cat_core_list = build_category_core_list(
            team_list, set_list, category_dict, config, team_preview
        )

        archetype_analyzer = ArchetypeAnalyzer(config)
        # Create plot output directory
        plot_dir = os.path.join('output', gen, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        result = archetype_analyzer.analyze(
            cat_core_list,
            save_plot=os.path.join(plot_dir, 'fpc_plot.png')
        )

        if result:
            output_writer.write_archetype_statistics(
                gen, archetype_analyzer, cat_core_list, category_nics, team_list
            )

    # Write output files
    output_writer.write_statistics_txt(
        gen, core_list, mpmi_list, lead_list, team_list,
        team_preview, True, mon_frequency
    )
    output_writer.write_statistics_csv(
        gen, core_list, mpmi_list, lead_list, team_list,
        team_preview, True, mon_frequency
    )

    # Write sets files
    for frac in config.ignore_sets_fraction:
        output_writer.write_sets_file(gen, set_list_moves2_sorted, mon_frequency, move_frequency, frac)

    # Sort and write builder
    if config.sort_builder:
        sort_teams(team_list, set_list, core_list, lead_list, folder_count,
                   config, team_preview, archetype_analyzer)
        output_writer.write_sorted_builder(gen, team_list, set_list, move_frequency, archetype_analyzer)

    return len(team_list)


def generate_category_nicknames(category_dict: dict, config: Config) -> dict:
    """Generate human-readable nicknames for categories."""
    category_nics = {}

    for name in category_dict:
        for cat in category_dict[name]:
            if cat == 'Count':
                continue

            full_cat = (name, cat)

            if cat in config.important_items or (
                isinstance(cat, str) and len(cat) > 0 and cat[-1] == 'Z' and 'Z' in config.important_items
            ):
                short_cat = str(cat).replace(' Berry', '').replace('Choice ', '').replace('Assault Vest', 'AV')
                category_nics[full_cat] = f"{short_cat} {name}"
            elif isinstance(cat, tuple):
                stat_str = f"{INDEX_TO_STAT[cat[0]]}/{INDEX_TO_STAT[cat[1]]}"
                category_nics[full_cat] = f"{stat_str} {name}"
            else:
                category_nics[full_cat] = name

    return category_nics


def build_category_core_list(team_list: List[dict], set_list: List[dict],
                             category_dict: dict, config: Config,
                             team_preview: bool) -> List[dict]:
    """Build core list based on categories for archetype analysis."""
    cat_core_list: List[Dict[Any, int]] = [{} for _ in range(6)]

    for team in team_list:
        if team['Anomalies'] > config.anomaly_threshold:
            continue

        category_list = []
        for s in set_list[team['Index'][0]:team['Index'][1]]:
            category = get_set_category(s, category_dict, config)
            if category:
                category_list.append(category)

        for p in range(min(6, len(category_list))):
            for c in combinations(category_list, p + 1):
                c_sort = tuple(sorted(c, key=lambda x: x[0]))
                cat_core_list[p][c_sort] = cat_core_list[p].get(c_sort, 0) + 1

    return cat_core_list


def get_set_category(s: dict, category_dict: dict, config: Config) -> tuple:
    """Determine the category tuple for a set."""
    category = [s['Name']]

    if s['Item'] in config.important_items or (
        len(s['Item']) > 0 and s['Item'][-1] == 'Z' and 'Z' in config.important_items
    ):
        category.append(s['Item'])
    else:
        current_evs = s['EVs'].copy()
        nature = s['Nature']
        if nature and nature in NATURE_DICT:
            modifier = NATURE_DICT[nature]
            if max(modifier) == 1:
                current_evs[modifier.index(1)] += config.nature_ev_modifier
                current_evs[modifier.index(-1)] -= config.nature_ev_modifier

        highest1 = current_evs.index(max(current_evs))
        temp = current_evs.copy()
        del temp[highest1]
        highest2 = temp.index(max(temp))
        if highest2 >= highest1:
            highest2 += 1
        category.append(tuple(sorted([highest1, highest2])))

    return tuple(category)


def sort_teams(team_list: List[dict], set_list: List[dict], core_list: List[dict],
               lead_list: dict, folder_count: dict, config: Config,
               team_preview: bool, archetype_analyzer) -> None:
    """Sort teams according to configuration."""

    def ord_string(s: str, reverse: bool) -> str:
        if not reverse:
            return s
        return ''.join(chr(1114111 - ord(c)) for c in s)

    def to_bool(n: int) -> bool:
        return n < 0

    def sort_key(team):
        key_list = []

        if config.sort_folder_by_frequency or config.sort_folder_by_alphabetical:
            if config.sort_folder_by_frequency:
                key_list.append(config.sort_folder_by_frequency * folder_count.get(team['Folder'], 0))
            key_list.append(ord_string(team['Folder'].casefold(), to_bool(config.sort_folder_by_alphabetical)))

        lead_freq_setting = (
            config.sort_teams_by_lead_frequency_team_preview if team_preview
            else config.sort_teams_by_lead_frequency_no_team_preview
        )

        if lead_freq_setting:
            lead_name = set_list[team['Index'][0]]['Name']
            if team_preview:
                key_list.append(lead_freq_setting * core_list[0].get((lead_name,), 0))
            else:
                key_list.append(lead_freq_setting * lead_list.get(lead_name, 0))
            key_list.append(ord_string(lead_name, False))

        if config.sort_teams_by_core:
            key_list.append(config.sort_teams_by_core * team['Score'][config.core_number - 1])

        if config.sort_teams_by_alphabetical:
            key_list.append(ord_string(team['Name'].casefold(), to_bool(config.sort_teams_by_alphabetical)))

        return tuple(key_list)

    team_list.sort(key=sort_key)


def main():
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print welcome banner
    console.print(Panel.fit(
        "[bold blue]BuilderAnalyzer[/bold blue]\n"
        "[dim]Pokemon Team Builder Analysis Tool[/dim]",
        border_style="blue"
    ))

    # Initialize configuration
    config = Config()
    if args.input:
        config.fin = args.input
    if args.no_download:
        config.download_pokedex = False
    if args.no_archetypes:
        config.analyze_archetypes = False

    force_download = args.force_download

    # Initialize data loader
    data_loader = DataLoader()

    if config.download_pokedex:
        with console.status("[bold green]Checking Pokemon data files..."):
            data_loader.download_data(force=force_download)

    with console.status("[bold green]Loading Pokemon data files..."):
        data_loader.load_data(load_colors=config.sort_mons_by_color)

    pokedex_str, items_str, moves_str, abilities_str, colors = data_loader.get_data()

    # Verify data loaded successfully
    if pokedex_str is None or items_str is None or moves_str is None or abilities_str is None:
        console.print("[bold red]Error:[/bold red] Failed to load Pokemon data files")
        sys.exit(1)

    # Initialize parser
    set_parser = SetParser(pokedex_str, items_str, abilities_str, moves_str)

    # Detect input format
    is_dense_format = detect_input_format(config.fin)

    # Get generations to process
    if config.all_generations:
        generations = get_generations_from_file(config.fin, is_dense_format)
    else:
        generations = config.generation

    console.print(f"[cyan]Found {len(generations)} generations to process[/cyan]")

    # Initialize output writer
    output_template = config.get_output_template()
    output_writer = OutputWriter(config, output_template)

    # Track stats
    num_teams_by_gen: Dict[str, int] = {}
    all_incomplete_sets: List[dict] = []
    all_incomplete_teams: List[dict] = []
    move_frequency: Dict[str, Dict[str, int]] = {}

    # Process each generation with progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[green]Processing generations...", total=len(generations))

        for gen in generations:
            progress.update(task, description=f"[green]Processing {gen}...")
            num_teams = process_generation(
                gen, config, data_loader, set_parser, output_writer,
                is_dense_format, all_incomplete_sets, all_incomplete_teams
            )
            num_teams_by_gen[gen] = num_teams
            progress.advance(task)

    # Write incomplete teams
    if all_incomplete_teams:
        output_writer.write_incomplete_teams(all_incomplete_teams, all_incomplete_sets, move_frequency)

    # Write full combined builder
    if config.sort_builder:
        # Sort generations
        if config.sort_gen_by_frequency:
            generations.sort(key=lambda x: num_teams_by_gen.get(x, 0), reverse=(config.sort_gen_by_frequency < 0))
        elif config.sort_gen_by_alphabetical:
            generations.sort(reverse=(config.sort_gen_by_alphabetical < 0))

        output_writer.write_full_builder(generations, config.include_incomplete_teams)

    # Print summary table
    total_teams = sum(num_teams_by_gen.values())
    summary = Table(title="Processing Complete", show_header=True, header_style="bold magenta")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="green")
    summary.add_row("Generations processed", str(len(generations)))
    summary.add_row("Total teams", str(total_teams))
    summary.add_row("Output directory", "output/")
    console.print(summary)

    # Clean up matplotlib resources
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
    except ImportError:
        pass

    # Flush and close log handlers
    logging.shutdown()


if __name__ == '__main__':
    main()
    sys.exit(0)
