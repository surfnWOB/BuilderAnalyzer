"""
Team analyzer module for BuilderAnalyzer.
Handles team extraction and completeness analysis.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from itertools import combinations
import copy
import logging

from .set_parser import SetParser

logger = logging.getLogger(__name__)


@dataclass
class Team:
    """Represents a Pokemon team."""
    name: str
    gen: str
    folder: str = ''
    index: Tuple[int, int] = (0, 0)  # Start and end indices in set list
    score: List[float] = field(default_factory=lambda: [0.0] * 6)
    line: int = 0
    anomalies: int = 0
    categories: List[tuple] = field(default_factory=list)


class TeamExtractor:
    """Extracts teams and sets from builder files."""

    def __init__(self, set_parser: SetParser):
        self.set_parser = set_parser

    def extract_teams(self, filename: str, gen: str, is_dense_format: bool) -> Tuple[List[dict], List[dict], Dict[str, int]]:
        """
        Extract all teams and sets for a given generation.

        Args:
            filename: Path to the builder file
            gen: Generation string to extract
            is_dense_format: Whether the file is in dense format

        Returns:
            Tuple of (set_list, team_list, folder_count)
        """
        logger.info(f"Extracting teams for {gen} from {filename}...")

        set_list: List[dict] = []
        team_list: List[dict] = []
        folder_count: Dict[str, int] = {'': 1}

        with open(filename, encoding='utf-8', errors='ignore') as f:
            line = f.readline()
            line_count = 0

            if is_dense_format:
                set_list, team_list, folder_count = self._extract_dense(f, line, gen, line_count)
            else:
                set_list, team_list, folder_count = self._extract_readable(f, line, gen, line_count)

        # Finalize last team index
        if team_list:
            team_list[-1]['Index'] = (team_list[-1]['Index'][0], len(set_list))
            if len(set_list) == team_list[-1]['Index'][0]:
                team_list.pop()

        logger.info(f"Extracted {len(team_list)} teams with {len(set_list)} sets for {gen}")
        return set_list, team_list, folder_count

    def _extract_dense(self, f: Any, line: str, gen: str, line_count: int) -> Tuple[List[dict], List[dict], Dict[str, int]]:
        """Extract teams from dense format file."""
        set_list: List[dict] = []
        team_list: List[dict] = []
        folder_count: Dict[str, int] = {'': 1}

        while line:
            line_count += 1
            idx_vert = line.find('|')
            idx_right = line.find(']')

            if line[0:3] == 'gen' and line.find(gen + ']') > -1:
                if team_list:
                    team_list[-1]['Index'] = (team_list[-1]['Index'][0], len(set_list))

                team_name = line[1 + len(gen):idx_vert]
                folder = ''

                if line.find('/') > -1:
                    folder = line[1 + len(gen):line.find('/') + 1]
                    team_name = line[line.find('/') + 1:idx_vert]
                    folder_count[folder] = folder_count.get(folder, 0) + 1

                team_list.append({
                    'Index': (len(set_list), len(set_list)),
                    'Name': team_name,
                    'Gen': gen,
                    'Folder': folder,
                    'Score': [0] * 6,
                    'Line': line_count,
                    'Anomalies': 0
                })

                # Parse sets from the line
                idx_right = idx_vert
                while idx_right > -1 and idx_right < len(line) - 2:
                    idx_right2 = line.find(']', idx_right + 1)
                    set_text = line[idx_right + 1:idx_right2]
                    parsed_set = self.set_parser.parse_set(set_text, True)
                    set_list.append(parsed_set)
                    idx_right = idx_right2

            line = f.readline()

        return set_list, team_list, folder_count

    def _extract_readable(self, f: Any, line: str, gen: str, line_count: int) -> Tuple[List[dict], List[dict], Dict[str, int]]:
        """Extract teams from readable format file."""
        set_list: List[dict] = []
        team_list: List[dict] = []
        folder_count: Dict[str, int] = {'': 1}
        buffer = line
        line_status = 0  # 0 = importable not found, 1 = found
        right_gen = False

        while line:
            line_count += 1

            if line.find('===') > -1 and line[0:3] == '===' and line[-4:-1] == '===':
                if line.find('[' + gen + ']') > -1:
                    if team_list:
                        team_list[-1]['Index'] = (team_list[-1]['Index'][0], len(set_list))

                    team_name = line[7 + len(gen):-5]
                    folder = ''

                    if line.find('/') > -1:
                        folder = line[7 + len(gen):line.find('/') + 1]
                        team_name = line[line.find('/') + 1:-5]
                        folder_count[folder] = folder_count.get(folder, 0) + 1

                    team_list.append({
                        'Index': (len(set_list), len(set_list)),
                        'Gen': gen,
                        'Name': team_name,
                        'Folder': folder,
                        'Score': [0] * 6,
                        'Line': line_count,
                        'Anomalies': 0
                    })
                    right_gen = True
                else:
                    right_gen = False
                    line = f.readline()
                    continue
            elif not right_gen:
                line = f.readline()
                continue

            # Look for set markers
            if line_status == 0:
                mark = self._find_set_marker(line)
                if mark == -1 and buffer == '\n':
                    mark = -1
                if mark == 0:
                    line_status = 1
                    mon_txt = buffer
                buffer = line
            if line_status == 1:
                if line == '\n':
                    parsed_set = self.set_parser.parse_set(mon_txt, False)
                    set_list.append(parsed_set)
                    line_status = 0
                else:
                    mon_txt = mon_txt + line

            line = f.readline()

        # Handle last set
        if line_status == 1:
            parsed_set = self.set_parser.parse_set(mon_txt + '  \n', False)
            set_list.append(parsed_set)

        return set_list, team_list, folder_count

    def _find_set_marker(self, line: str) -> int:
        """Find a marker indicating the start of a set."""
        markers = ['Ability:', 'Level: ', 'Shiny: ', 'EVs: ', 'IVs: ']
        for marker in markers:
            mark = line.find(marker)
            if mark > -1:
                return mark if marker != 'Ability:' else 0

        mark = line.find('Nature  \n')
        if mark > -1:
            return 0

        mark = line.find('-')
        return mark


def check_team_completeness(team_list: List[dict], set_list: List[dict], gen: str,
                            anomaly_threshold: int) -> Tuple[List[dict], List[dict]]:
    """
    Check which teams are complete and separate incomplete ones.

    Args:
        team_list: List of team dictionaries
        set_list: List of set dictionaries
        gen: Generation string
        anomaly_threshold: Threshold for anomaly count

    Returns:
        Tuple of (incomplete_sets, incomplete_teams)
    """
    logger.info(f"Checking team completeness for {gen}...")

    # Determine max mons based on format
    if gen.find('1v1') > 0:
        max_mons = 1
    elif gen.find('metronome') > 0:
        max_mons = 2
    else:
        max_mons = 6

    set_list_incomplete: List[dict] = []
    team_list_incomplete: List[dict] = []

    for n, team in enumerate(team_list):
        team['Anomalies'] = 0

        # Check if team has fewer Pokemon than expected
        team_size = team['Index'][1] - team['Index'][0]
        if team_size < max_mons:
            team['Anomalies'] += 6

        # Check individual sets
        for s in set_list[team['Index'][0]:team['Index'][1]]:
            gen_num = int(gen[3]) if gen[3].isdigit() else 9

            # Check EVs (gen 3+ only, excluding Let's Go)
            if gen_num > 2 and gen.find('letsgo') == -1:
                ev_total = sum(s['EVs'])
                ev_limit = 508 - (400 / s['Level'])
                if ev_total < ev_limit:
                    team['Anomalies'] += 1

            # Check moves
            if len(s['Moveset']) < 4 and s['Name'] != 'Ditto':
                team['Anomalies'] += 1

        # Copy incomplete teams
        if team['Anomalies'] > anomaly_threshold:
            incomplete_team = copy.deepcopy(team)
            incomplete_team['Index'] = (len(set_list_incomplete), len(set_list_incomplete))
            set_list_incomplete.extend(copy.deepcopy(set_list[team['Index'][0]:team['Index'][1]]))
            incomplete_team['Index'] = (incomplete_team['Index'][0], len(set_list_incomplete))
            team_list_incomplete.append(incomplete_team)

    complete_count = sum(1 for t in team_list if t['Anomalies'] <= anomaly_threshold)
    logger.info(f"Found {complete_count} complete teams, {len(team_list_incomplete)} incomplete")

    return set_list_incomplete, team_list_incomplete


def find_cores_and_leads(team_list: List[dict], set_list: List[dict],
                         anomaly_threshold: int, team_preview: bool,
                         show_missing_mon_cores: bool, max_missing_mon_cores: int) -> Tuple[List[dict], dict]:
    """
    Find core combinations and lead Pokemon statistics.

    Args:
        team_list: List of team dictionaries
        set_list: List of set dictionaries
        anomaly_threshold: Threshold for including teams
        team_preview: Whether this is a team preview format
        show_missing_mon_cores: Whether to include missing cores
        max_missing_mon_cores: Maximum core size for missing cores

    Returns:
        Tuple of (core_list, lead_list)
    """
    logger.info("Finding cores and leads...")

    core_list: List[Dict[Tuple[str, ...], int]] = [{} for _ in range(6)]
    lead_list: Dict[str, int] = {}

    for team in team_list:
        inc = 0 if team['Anomalies'] > anomaly_threshold else 1

        for p in range(6):
            mons = [s['Name'] for s in set_list[team['Index'][0]:team['Index'][1]]]
            for c in combinations(mons, p + 1):
                c_sort = tuple(sorted(c))
                core_list[p][c_sort] = core_list[p].get(c_sort, 0) + inc

        if not team_preview and team['Index'][1] > team['Index'][0]:
            lead_name = set_list[team['Index'][0]]['Name']
            lead_list[lead_name] = lead_list.get(lead_name, 0) + inc

    # Add missing cores if requested
    if show_missing_mon_cores:
        for p in range(1, max_missing_mon_cores):
            all_mons = [c[0] for c in core_list[0]]
            for c in combinations(all_mons, p + 1):
                c_sort = tuple(sorted(c))
                if c_sort not in core_list[p]:
                    core_list[p][c_sort] = 0

    # Log stats
    for p in range(6):
        logger.debug(f"{p+1}-cores: {len(core_list[p])} combinations")

    return core_list, lead_list
