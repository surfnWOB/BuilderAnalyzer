"""
Statistics module for BuilderAnalyzer.
Handles MPMI calculations, set categorization, and aggregation.
"""

from itertools import combinations
from typing import List, Dict, Tuple, Any
import copy
import math
import logging

from .config import Config, NATURE_DICT

logger = logging.getLogger(__name__)


def calculate_mpmi(core_list: List[dict], core_count: List[int],
                   multiplicity: List[int], show_missing: bool = False) -> List[dict]:
    """
    Calculate multivariate pointwise mutual information for cores.

    Args:
        core_list: List of core dictionaries by size
        core_count: Total counts for each core size
        multiplicity: Factorial multipliers for each size
        show_missing: Whether to include missing cores

    Returns:
        List of MPMI dictionaries by core size
    """
    logger.info("Calculating MPMI for cores...")
    mpmi_list: List[Dict[Tuple[str, ...], float]] = [{} for _ in range(6)]

    for p in range(6):
        for c in core_list[p]:
            if core_list[p][c] != 0:
                mpmi_list[p][c] = 0
                for q in range(p + 1):
                    for d in combinations(c, q + 1):
                        d_sort = tuple(sorted(d))
                        if d_sort in core_list[q] and core_list[q][d_sort] != 0:
                            # Sign convention: positive if synergistic
                            mpmi_list[p][c] += ((-1) ** q) * (
                                -math.log(core_list[q][d_sort] / core_count[q] / multiplicity[q], 2)
                            )
                        else:
                            mpmi_list[p][c] = 0
            elif show_missing:
                # Handle missing cores
                mpmi_list[p][c] = -100
                for m in c:
                    if (m,) in core_list[0] and core_list[0][(m,)] > 0:
                        mpmi_list[p][c] += -math.log(
                            core_list[0][(m,)] / core_count[0] / multiplicity[0], 2
                        )

    return mpmi_list


def categorize_sets(set_list: List[dict], config: Config) -> dict:
    """
    Categorize Pokemon sets by item/EVs and analyze move patterns.

    Args:
        set_list: List of Pokemon sets sorted by name
        config: Configuration object

    Returns:
        Dictionary of categories by Pokemon name
    """
    logger.info("Categorizing sets...")
    category_dict: Dict[str, Dict[Any, Any]] = {}

    for current_set in set_list:
        name = current_set['Name']

        if name not in category_dict:
            category_dict[name] = {'Count': 0}
        category_dict[name]['Count'] += 1  # type: ignore[operator]

        # Determine category based on item or EVs
        category1 = _determine_category(current_set, config)

        if category1 not in category_dict[name]:
            category_dict[name][category1] = _create_category_entry()
        category_dict[name][category1]['Count'] += 1  # type: ignore[index]

        # Generate move statistics
        for p in range(3):
            for c in combinations(current_set['Moveset'], p + 1):
                c_sort = tuple(sorted(c))
                counts = category_dict[name][category1]['ActualCount'][p]  # type: ignore[index]
                counts[c_sort] = counts.get(c_sort, 0) + 1
                category_dict[name][category1]['TotalCount'][p] += 1  # type: ignore[index]

    # Calculate move synergies
    _calculate_move_synergies(category_dict, config)

    logger.info(f"Categorized {len(category_dict)} Pokemon")
    return category_dict


def _determine_category(current_set: dict, config: Config) -> Any:
    """Determine the category for a set based on item or EVs."""
    item = current_set['Item']

    # Check for important items
    if item in config.important_items:
        return item
    if len(item) > 0 and item[-1] == 'Z' and 'Z' in config.important_items:
        return item

    # Use EVs for categorization
    current_evs = copy.deepcopy(current_set['EVs'])
    nature = current_set['Nature']

    if nature and nature in NATURE_DICT:
        modifier = NATURE_DICT[nature]
        if max(modifier) == 1:
            increase_stat = modifier.index(1)
            decrease_stat = modifier.index(-1)
            current_evs[increase_stat] += config.nature_ev_modifier
            current_evs[decrease_stat] -= config.nature_ev_modifier

    # Find top two EV stats
    highest1 = current_evs.index(max(current_evs))
    temp_evs = current_evs.copy()
    del temp_evs[highest1]
    highest2 = temp_evs.index(max(temp_evs))
    if highest2 >= highest1:
        highest2 += 1

    return tuple(sorted([highest1, highest2]))


def _create_category_entry() -> dict:
    """Create a new category entry with default structure."""
    return {
        'ActualCount': [{}, {}, {}],
        'PriorProb': [{}, {}],
        'ActualProb': [{}, {}],
        'SumProb': [{}, {}],
        'MutualInfo': [{}, {}],
        'TotalCount': [0, 0, 0],
        'Count': 0
    }


def _calculate_move_synergies(category_dict: dict, config: Config) -> None:
    """Calculate move pair and triplet synergies for categorization."""
    for name in category_dict:
        for cat in category_dict[name]:
            if cat == 'Count' or cat in config.important_items:
                continue

            cat_data = category_dict[name][cat]
            total_move_count = cat_data['TotalCount'][0]
            total_pair_count = cat_data['TotalCount'][1]

            if total_move_count == 0:
                continue

            # Calculate move pair probabilities
            moves = [c[0] for c in cat_data['ActualCount'][0]]
            move_pair_candidates = []

            for c in combinations(moves, 2):
                c_sort = tuple(sorted(c))
                prob_move = [
                    cat_data['ActualCount'][0][(m,)] / total_move_count
                    for m in c_sort
                ]

                # Prior probability
                cat_data['PriorProb'][0][c_sort] = (
                    prob_move[0] * prob_move[1] *
                    (1 / (1 - prob_move[0]) + 1 / (1 - prob_move[1]))
                )

                # Actual probability
                if c_sort in cat_data['ActualCount'][1]:
                    cat_data['ActualProb'][0][c_sort] = (
                        cat_data['ActualCount'][1][c_sort] / total_pair_count
                    )
                else:
                    cat_data['ActualProb'][0][c_sort] = 0

                # Mutual information
                if cat_data['PriorProb'][0][c_sort] > 0:
                    cat_data['MutualInfo'][0][c_sort] = (
                        cat_data['ActualProb'][0][c_sort] / cat_data['PriorProb'][0][c_sort]
                    )
                else:
                    cat_data['MutualInfo'][0][c_sort] = 0

                # Sum probability
                cat_data['SumProb'][0][c_sort] = (
                    sum(prob_move) - cat_data['ActualProb'][0][c_sort]
                )

                # Check if this is a split candidate
                if cat_data['MutualInfo'][0][c_sort] < config.move_pair_synergy_threshold:
                    if min(prob_move) > (1/4) * config.move_prob_threshold:
                        min_count = min(
                            cat_data['ActualCount'][0][(m,)] for m in c_sort
                        )
                        if min_count > config.move_count_threshold:
                            move_pair_candidates.append(c_sort)

            # Choose best move pair split
            if move_pair_candidates:
                chosen = max(
                    move_pair_candidates,
                    key=lambda x: cat_data['SumProb'][0][x]
                )
                if cat_data['SumProb'][0][chosen] > (1/4) * config.sum_move_prob_threshold:
                    cat_data['SplitMoves'] = chosen


def aggregate_sets_by_ev(set_list: List[dict], config: Config) -> List[dict]:
    """
    Aggregate sets that differ only by small EV/IV differences.

    Args:
        set_list: Sorted list of Pokemon sets
        config: Configuration object

    Returns:
        List of aggregated sets
    """
    logger.info("Aggregating sets by EV similarity...")
    combined: List[dict] = []
    chosen_mon = ''
    chosen_mon_idx = 0

    for n, current_set in enumerate(set_list):
        current_count = 1
        match_index = -1
        nn = chosen_mon_idx

        while match_index < 0 and nn < len(combined):
            if current_set['Name'] == chosen_mon:
                if _sets_match_except_evs(current_set, combined[nn]):
                    ev_diff = sum(abs(a - b) for a, b in zip(
                        current_set['EVs'], combined[nn]['EVs']
                    ))
                    iv_diff = sum(abs(a - b) for a, b in zip(
                        current_set['IVs'], combined[nn]['IVs']
                    ))

                    if ev_diff <= config.ev_threshold * 2 and iv_diff <= config.iv_threshold:
                        match_index = nn
                        combined[nn]['CountEV'] += current_count
                        if current_count > combined[nn]['SubCountEV']:
                            combined[nn]['SubCountEV'] = current_count
                            combined[nn]['EVs'] = current_set['EVs']
                            combined[nn]['IVs'] = current_set['IVs']
                            combined[nn]['Index'] = n
            else:
                chosen_mon_idx = len(combined)
                chosen_mon = current_set['Name']
            nn += 1

        if n == 0:
            chosen_mon_idx = len(combined)
            chosen_mon = current_set['Name']

        if match_index < 0:
            new_set = copy.deepcopy(current_set)
            new_set['SubCountEV'] = current_count
            new_set['CountEV'] = current_count
            new_set['Index'] = n
            combined.append(new_set)

    logger.info(f"Aggregated to {len(combined)} unique EV sets")
    return combined


def _sets_match_except_evs(set1: dict, set2: dict) -> bool:
    """Check if two sets match except for EVs and IVs."""
    return (
        set1['Nature'] == set2['Nature'] and
        set1['Ability'] == set2['Ability'] and
        set1['Level'] == set2['Level'] and
        set1['Happiness'] == set2['Happiness'] and
        set1['Item'] == set2['Item'] and
        set(set1['Moveset']) == set(set2['Moveset'])
    )


def aggregate_sets_by_moves(set_list: List[dict], config: Config, slots: int = 1) -> List[dict]:
    """
    Aggregate sets that differ by a certain number of move slots.

    Args:
        set_list: List of Pokemon sets
        config: Configuration object
        slots: Number of move slots that can differ (1 or 2)

    Returns:
        List of aggregated sets
    """
    logger.info(f"Aggregating sets by {slots} move slot(s)...")
    combined: List[dict] = []
    chosen_mon = ''
    chosen_mon_idx = 0
    shared_key = 'SharedMoves1' if slots == 1 else 'SharedMoves2'

    for n, current_set in enumerate(set_list):
        current_set[shared_key] = {}
        current_count = current_set.get('CountEV', 1) if slots == 1 else current_set.get('CountMoves', 1)
        current_set['CountMoves'] = current_count

        match_index = -1
        nn = chosen_mon_idx

        while match_index < 0 and nn < len(combined) and config.combine_moves >= slots:
            if current_set['Name'] == chosen_mon:
                if _sets_match_for_move_combine(current_set, combined[nn], config):
                    match_result = _check_move_match(
                        current_set, combined[nn], current_count, shared_key, slots
                    )
                    if match_result:
                        match_index = nn
                        combined[nn]['CountMoves'] += current_count
            else:
                chosen_mon_idx = len(combined)
                chosen_mon = current_set['Name']
            nn += 1

        if n == 0:
            chosen_mon_idx = len(combined)
            chosen_mon = current_set['Name']

        if match_index < 0:
            combined.append(current_set)

    logger.info(f"Aggregated to {len(combined)} sets after {slots}-slot combining")
    return combined


def _sets_match_for_move_combine(set1: dict, set2: dict, config: Config) -> bool:
    """Check if sets match for move combination."""
    if not (
        set1['Nature'] == set2['Nature'] and
        set1['Ability'] == set2['Ability'] and
        set1['Level'] == set2['Level'] and
        set1['Happiness'] == set2['Happiness'] and
        set1['Item'] == set2['Item']
    ):
        return False

    ev_diff = sum(abs(a - b) for a, b in zip(set1['EVs'], set2['EVs']))
    iv_diff = sum(abs(a - b) for a, b in zip(set1['IVs'], set2['IVs']))

    return ev_diff <= config.ev_threshold * 2 and iv_diff <= config.iv_threshold


def _check_move_match(current_set: dict, combined_set: dict, current_count: int,
                      shared_key: str, slots: int) -> bool:
    """Check and update move matching for aggregation."""
    moves1 = current_set['Moveset']
    moves2 = combined_set['Moveset']

    if len(moves1) != len(moves2):
        return False

    common_moves1 = sum(1 for m in moves1 if m in moves2)
    common_moves2 = sum(1 for m in moves2 if m in moves1)

    if common_moves1 != common_moves2:
        return False

    # Check if exactly one move differs
    if common_moves1 + 1 != len(moves2):
        return False

    non_common1 = [m for m in moves1 if m not in moves2][0] if common_moves1 < len(moves1) else None
    non_common2 = [m for m in moves2 if m not in moves1][0] if common_moves2 < len(moves2) else None

    if slots == 1:
        if combined_set[shared_key]:
            if non_common2 in combined_set[shared_key]:
                combined_set[shared_key][non_common1] = current_count
                return True
        else:
            combined_set[shared_key][non_common2] = combined_set['CountMoves']
            combined_set[shared_key][non_common1] = current_count
            return True
    elif slots == 2:
        # For two-slot combining, also check SharedMoves1 compatibility
        if set(current_set.get('SharedMoves1', {}).keys()) == set(combined_set.get('SharedMoves1', {}).keys()):
            if combined_set[shared_key]:
                if non_common2 in combined_set[shared_key]:
                    combined_set[shared_key][non_common1] = current_set.get('SharedMoves1', {})
                    return True
            else:
                combined_set[shared_key][non_common2] = combined_set.get('SharedMoves1', {})
                combined_set[shared_key][non_common1] = current_set.get('SharedMoves1', {})
                return True

    return False


def sort_sets_by_name_and_count(set_list: List[dict]) -> List[dict]:
    """Sort sets by Pokemon name (alphabetical) and count (descending)."""
    def rank_key(s):
        name = s['Name']
        count = s.get('CountMoves', s.get('CountEV', 1))
        front1 = ord(name[0])
        front2 = ord(name[1]) if len(name) > 1 else 0
        sep_pos = max(name.find('-'), name.find(' '))
        if sep_pos > -1:
            back1 = ord(name[sep_pos + 1]) if sep_pos + 1 < len(name) else 0
            back2 = ord(name[sep_pos + 2]) if sep_pos + 2 < len(name) else 0
        else:
            back1 = ord(name[-2]) if len(name) > 1 else 0
            back2 = ord(name[-1])
        return (front1 * 2**24 + front2 * 2**16 + back1 * 2**8 + back2) * 2**10 - count

    return sorted(set_list, key=rank_key)


def count_move_frequency(set_list: List[dict]) -> Tuple[dict, dict]:
    """
    Count move and Pokemon frequencies from the set list.

    Returns:
        Tuple of (move_frequency, mon_frequency) dictionaries
    """
    move_frequency: Dict[str, Dict[str, int]] = {}
    mon_frequency: Dict[str, int] = {}

    for s in set_list:
        name = s['Name']
        count = s.get('CountEV', 1)

        if name not in move_frequency:
            move_frequency[name] = {}
            mon_frequency[name] = count
        else:
            mon_frequency[name] += count

        for m in s['Moveset']:
            move_frequency[name][m] = move_frequency[name].get(m, 0) + count

    return move_frequency, mon_frequency
