"""
Set parsing module for BuilderAnalyzer.
Handles parsing and printing of Pokemon sets in both dense and readable formats.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import logging

from .config import INDEX_TO_STAT

logger = logging.getLogger(__name__)


@dataclass
class PokemonSet:
    """Represents a single Pokemon set."""
    name: str = ''
    nickname: str = ''
    gender: str = ''
    item: str = ''
    ability: str = ''
    nature: str = ''
    shiny: bool = False
    moveset: List[str] = field(default_factory=list)
    evs: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 0, 0])
    ivs: List[int] = field(default_factory=lambda: [31, 31, 31, 31, 31, 31])
    level: int = 100
    happiness: int = 255
    alternate_form: str = ''
    shared_moves1: Dict[str, int] = field(default_factory=dict)
    shared_moves2: Dict[str, Dict[str, int]] = field(default_factory=dict)
    count_ev: int = 1
    sub_count_ev: int = 1
    count_moves: int = 1
    index: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary format for compatibility."""
        return {
            'Name': self.name,
            'Nickname': self.nickname,
            'Gender': self.gender,
            'Item': self.item,
            'Ability': self.ability,
            'Nature': self.nature,
            'Shiny': self.shiny,
            'Moveset': self.moveset.copy(),
            'EVs': self.evs.copy(),
            'IVs': self.ivs.copy(),
            'Level': self.level,
            'Happiness': self.happiness,
            'AlternateForm': self.alternate_form,
            'SharedMoves1': self.shared_moves1.copy(),
            'SharedMoves2': {k: v.copy() for k, v in self.shared_moves2.items()},
            'CountEV': self.count_ev,
            'SubCountEV': self.sub_count_ev,
            'CountMoves': self.count_moves,
            'Index': self.index,
        }


class SetParser:
    """Parses Pokemon sets from text."""

    def __init__(self, pokedex_str: str, items_str: str, abilities_str: str, moves_str: str):
        self.pokedex_str = pokedex_str
        self.items_str = items_str
        self.abilities_str = abilities_str
        self.moves_str = moves_str
        self.stat_to_index = {'HP': 0, 'Atk': 1, 'Def': 2, 'SpA': 3, 'SpD': 4, 'Spe': 5}

    def parse_set(self, text: str, is_dense_format: bool) -> dict:
        """
        Parse a Pokemon set from text.

        Args:
            text: The set text to parse
            is_dense_format: Whether the text is in dense (pipe-delimited) format

        Returns:
            dict: Parsed set data
        """
        if is_dense_format:
            return self._parse_dense(text)
        else:
            return self._parse_readable(text)

    def _parse_dense(self, text: str) -> dict:
        """Parse a set in dense pipe-delimited format."""
        set_dict = self._create_empty_set()

        # Parse nickname
        idx2 = text.find('|')
        set_dict['Nickname'] = text[0:idx2]

        # Parse name
        idx1 = idx2
        idx2 = text.find('|', idx1 + 1)
        if idx1 + 1 < idx2:
            parse_name = text[idx1 + 1:idx2]
            set_dict['Name'], set_dict['AlternateForm'] = self._lookup_pokemon_name(parse_name)
        else:
            set_dict['Name'] = set_dict['Nickname']
            set_dict['Nickname'] = ''
            if set_dict['Name'].find('-') > -1:
                idx_hyphen = set_dict['Name'].find('-')
                set_dict['AlternateForm'] = set_dict['Name'][idx_hyphen + 1:]

        # Parse item
        idx1 = idx2
        idx2 = text.find('|', idx1 + 1)
        if idx1 + 1 < idx2:
            parse_item = text[idx1 + 1:idx2]
            set_dict['Item'] = self._lookup_item_name(parse_item)

        # Parse ability
        idx1 = idx2
        idx2 = text.find('|', idx1 + 1)
        parse_ability = text[idx1 + 1:idx2]
        set_dict['Ability'] = self._lookup_ability(parse_ability, set_dict['Name'], set_dict['AlternateForm'])

        # Parse moves
        idx1 = idx2
        idx2 = text.find('|', idx1 + 1)
        if idx1 + 1 < idx2:
            set_dict['Moveset'] = self._parse_moves(text[idx1 + 1:idx2])

        # Parse nature
        idx1 = idx2
        idx2 = text.find('|', idx1 + 1)
        if idx1 + 1 < idx2:
            set_dict['Nature'] = text[idx1 + 1:idx2]

        # Parse EVs
        idx1 = idx2
        idx2 = text.find('|', idx1 + 1)
        if idx1 + 2 < idx2:
            set_dict['EVs'] = self._parse_stats(text[idx1 + 1:idx2], default=0)

        # Parse gender
        idx1 = idx2
        idx2 = text.find('|', idx1 + 1)
        if idx1 + 1 < idx2:
            set_dict['Gender'] = text[idx1 + 1:idx2]

        # Parse IVs
        idx1 = idx2
        idx2 = text.find('|', idx1 + 1)
        if idx1 + 1 < idx2:
            set_dict['IVs'] = self._parse_stats(text[idx1 + 1:idx2], default=31)

        # Parse shiny
        idx1 = idx2
        idx2 = text.find('|', idx1 + 1)
        if idx1 + 1 < idx2:
            set_dict['Shiny'] = (text[idx1 + 1:idx2] == 'S')

        # Parse level
        idx1 = idx2
        idx2 = text.find('|', idx1 + 1)
        if idx1 + 1 < idx2:
            level = text[idx1 + 1:idx2]
            set_dict['Level'] = int(level) if level else 100

        # Parse happiness
        idx1 = idx2
        happiness = text[idx1 + 1:]
        if happiness.strip():
            try:
                set_dict['Happiness'] = int(happiness)
            except ValueError:
                pass

        # Handle mega evolution
        self._handle_mega_evolution(set_dict)

        return set_dict

    def _parse_readable(self, text: str) -> dict:
        """Parse a set in readable newline format."""
        set_dict = self._create_empty_set()

        # Parse name and gender
        pos2 = max(text.rfind(' (F) @ '), text.rfind(' (F)  \n'))
        if pos2 > -1:
            set_dict['Gender'] = 'F'
        else:
            pos2 = max(text.rfind(' (M) @ '), text.rfind(' (M)  \n'))
            if pos2 > -1:
                set_dict['Gender'] = 'M'
            else:
                pos2 = text.rfind(' @ ')
                if pos2 == -1:
                    pos2 = text.find('  \n')

        if text[pos2 - 1] == ')':
            pos2 = pos2 - 1
            pos1 = pos2 - 1
            while text[pos1] != '(':
                pos1 = pos1 - 1
            pos1 = pos1 + 1
            set_dict['Nickname'] = text[:pos1 - 2]
        else:
            pos1 = 0

        pos_item = text.find(' @ ', pos2) + 3
        set_dict['Name'] = text[pos1:pos2]
        if pos_item != -1:
            set_dict['Item'] = text[pos_item:text.find('  \n')]

        # Parse shiny
        set_dict['Shiny'] = text.find('\nShiny: Yes') > -1

        # Parse moves
        pos_move1 = len(text)
        while text.rfind('- ', 0, pos_move1) > -1:
            pos_move1 = text.rfind('- ', 0, pos_move1)
            pos_move2 = text.find('\n', pos_move1)
            set_dict['Moveset'].insert(0, text[pos_move1 + 2:pos_move2 - 2])

        # Parse EVs
        if text.find('\nEVs: ') > -1:
            ev_start = text.find('\nEVs: ') + 5
            ev_end = text.find('\n', ev_start)
            set_dict['EVs'] = self._parse_readable_stats(text[ev_start:ev_end], default=0)

        # Parse IVs
        if text.find('\nIVs: ') > -1:
            iv_start = text.find('\nIVs: ') + 5
            iv_end = text.find('\n', iv_start)
            set_dict['IVs'] = self._parse_readable_stats(text[iv_start:iv_end], default=31)

        # Parse nature
        pos_nature2 = text.find('Nature  \n') - 1
        if pos_nature2 > -1:
            pos_nature1 = text.rfind('\n', 0, pos_nature2) + 1
            set_dict['Nature'] = text[pos_nature1:pos_nature2]

        # Parse ability
        pos_ability1 = text.find('\nAbility: ') + 10
        pos_ability2 = text.find('\n', pos_ability1) - 2
        set_dict['Ability'] = text[pos_ability1:pos_ability2]

        # Parse level
        if text.find('\nLevel: ') > -1:
            pos_level1 = text.find('\nLevel: ') + 8
            pos_level2 = text.find('\n', pos_level1)
            set_dict['Level'] = int(text[pos_level1:pos_level2])

        # Parse happiness
        if text.find('\nHappiness: ') > -1:
            pos_happy1 = text.find('\nHappiness: ') + 12
            pos_happy2 = text.find('\n', pos_happy1)
            set_dict['Happiness'] = int(text[pos_happy1:pos_happy2])

        # Parse item (alternative location)
        pos_temp = text.find(set_dict['Name']) + len(set_dict['Name'])
        if text.find(' @ ', pos_temp) > -1:
            pos_item1 = text.find(' @ ', pos_temp) + 3
            pos_item2 = text.find('\n', pos_item1) - 2
            set_dict['Item'] = text[pos_item1:pos_item2]

        # Handle mega evolution
        self._handle_mega_evolution(set_dict)

        return set_dict

    def _create_empty_set(self) -> dict:
        """Create an empty set dictionary with default values."""
        return {
            'Name': '',
            'Nickname': '',
            'Gender': '',
            'Item': '',
            'Ability': '',
            'Nature': '',
            'Shiny': False,
            'Moveset': [],
            'EVs': [0, 0, 0, 0, 0, 0],
            'IVs': [31, 31, 31, 31, 31, 31],
            'Level': 100,
            'Happiness': 255,
            'AlternateForm': '',
            'SharedMoves1': {},
            'SharedMoves2': {},
        }

    def _lookup_pokemon_name(self, parse_name: str) -> tuple:
        """Look up the proper Pokemon name from the pokedex."""
        # Use tab prefix to match exact key (prevents substring matches)
        idx_name_key = self.pokedex_str.find('\t' + parse_name + ': ')
        alternate_form = ''

        if idx_name_key == -1:
            # Might be an alternate form - still use tab prefix
            idx_name_key = self.pokedex_str.find('\t' + parse_name)
            if idx_name_key == -1:
                logger.warning(f"Pokemon name '{parse_name}' not found")
                return parse_name, ''

            idx_species = self.pokedex_str.rfind('name: ', 0, idx_name_key)
            idx_base2 = self.pokedex_str.rfind('{', 0, idx_species) - 2
            idx_base1 = self.pokedex_str.rfind('\t', 0, idx_base2) + 1
            name_key_base = self.pokedex_str[idx_base1:idx_base2]
            alternate_form = parse_name[len(name_key_base):].capitalize()
        else:
            idx_species = self.pokedex_str.find('name: ', idx_name_key)

        idx_name1 = self.pokedex_str.find('"', idx_species)
        idx_name2 = self.pokedex_str.find('"', idx_name1 + 1)
        name = self.pokedex_str[idx_name1 + 1:idx_name2]

        if alternate_form:
            name = f"{name}-{alternate_form}"

        return name, alternate_form

    def _lookup_item_name(self, parse_item: str) -> str:
        """Look up the proper item name from the items data."""
        # Use tab prefix to match exact key (prevents substring matches)
        idx_item_key = self.items_str.find('\t' + parse_item + ': {')
        if idx_item_key == -1:
            return parse_item

        idx_item_name = self.items_str.find('name: ', idx_item_key)
        idx1 = self.items_str.find('"', idx_item_name)
        idx2 = self.items_str.find('"', idx1 + 1)
        return self.items_str[idx1 + 1:idx2]

    def _lookup_ability(self, parse_ability: str, pokemon_name: str, alternate_form: str) -> str:
        """Look up the ability name, handling slot references."""
        if not parse_ability:
            parse_ability = '0'

        if len(parse_ability) <= 1:
            # It's an ability slot reference, look it up in pokedex
            idx_name = self.pokedex_str.find('"' + pokemon_name.split('-')[0] + '"')
            if idx_name == -1:
                return parse_ability

            if alternate_form:
                idx_abilities = self.pokedex_str.rfind('abilities: ', 0, idx_name)
            else:
                idx_abilities = self.pokedex_str.find('abilities: ', idx_name)

            idx_ability = self.pokedex_str.find(parse_ability + ': ', idx_abilities)
            idx1 = self.pokedex_str.find('"', idx_ability)
            idx2 = self.pokedex_str.find('"', idx1 + 1)
            return self.pokedex_str[idx1 + 1:idx2]
        elif parse_ability == 'none':
            return 'none'
        else:
            # Use tab prefix to match exact key (prevents substring matches)
            idx_ability_key = self.abilities_str.find('\t' + parse_ability + ': {')
            if idx_ability_key == -1:
                return parse_ability

            idx_ability = self.abilities_str.find('name: ', idx_ability_key)
            idx1 = self.abilities_str.find('"', idx_ability)
            idx2 = self.abilities_str.find('"', idx1 + 1)
            return self.abilities_str[idx1 + 1:idx2]

    def _parse_moves(self, text: str) -> List[str]:
        """Parse moves from comma-separated dense format."""
        moves = []
        idx1 = -1
        while idx1 < len(text):
            idx2 = text.find(',', idx1 + 1)
            if idx2 == -1:
                idx2 = len(text)

            parse_move = text[idx1 + 1:idx2]
            # Use tab prefix to match exact move key (prevents 'roar' matching 'nobleroar')
            idx_move_key = self.moves_str.find('\t' + parse_move + ': {')
            if idx_move_key != -1:
                idx_move_name = self.moves_str.find('name: ', idx_move_key)
                idx_name1 = self.moves_str.find('"', idx_move_name)
                idx_name2 = self.moves_str.find('"', idx_name1 + 1)
                moves.append(self.moves_str[idx_name1 + 1:idx_name2])

            idx1 = idx2

        return moves

    def _parse_stats(self, text: str, default: int) -> List[int]:
        """Parse stats from comma-separated format."""
        stats = [default] * 6
        parts = text.split(',')
        for i, part in enumerate(parts[:6]):
            if part.strip():
                try:
                    stats[i] = int(part)
                except ValueError:
                    stats[i] = default
        return stats

    def _parse_readable_stats(self, text: str, default: int) -> List[int]:
        """Parse stats from readable 'X Stat / Y Stat' format."""
        stats = [default] * 6
        parts = text.split('/')
        for part in parts:
            part = part.strip()
            space_idx = part.find(' ')
            if space_idx > 0:
                try:
                    value = int(part[:space_idx])
                    stat_name = part[space_idx + 1:].strip()
                    if stat_name in self.stat_to_index:
                        stats[self.stat_to_index[stat_name]] = value
                except ValueError:
                    pass
        return stats

    def _handle_mega_evolution(self, set_dict: dict) -> None:
        """Handle mega evolution based on held item."""
        item = set_dict['Item']
        idx_mega_stone = item.find('ite')

        if idx_mega_stone > -1:
            if idx_mega_stone == len(item) - 3 and item != 'Eviolite':
                if set_dict['Name'].find('-Mega') == -1:
                    set_dict['Name'] = set_dict['Name'] + '-Mega'
            elif idx_mega_stone == len(item) - 5:
                if set_dict['Name'] == 'Charizard':
                    set_dict['Name'] = 'Charizard-Mega-' + item[-1]


def print_set(set_dict: dict, move_frequency: dict, show_shiny: bool, show_ivs: bool,
              show_nicknames: bool, sort_moves_alphabetical: int, sort_moves_frequency: int) -> str:
    """
    Format a Pokemon set for text output.

    Args:
        set_dict: The set dictionary to format
        move_frequency: Dictionary of move frequencies by Pokemon
        show_shiny: Whether to show shiny status
        show_ivs: Whether to show IVs
        show_nicknames: Whether to show nicknames
        sort_moves_alphabetical: Sort moves alphabetically (-1, 0, 1)
        sort_moves_frequency: Sort moves by frequency (-1, 0, 1)

    Returns:
        str: Formatted set text
    """
    text = ''

    # Name and nickname
    if set_dict['Nickname'] and show_nicknames:
        text += f"{set_dict['Nickname']} ("
    text += set_dict['Name']
    if set_dict['Nickname'] and show_nicknames:
        text += ')'

    # Gender
    if set_dict['Gender']:
        text += f" ({set_dict['Gender']})"

    # Item
    if set_dict['Item']:
        text += f" @ {set_dict['Item']}"

    # Ability
    text += f"  \nAbility: {set_dict['Ability']}"

    # Level
    if set_dict['Level'] != 100:
        text += f"  \nLevel: {int(set_dict['Level'])}"

    # Shiny
    if set_dict['Shiny'] and show_shiny:
        text += ' \nShiny: Yes'

    # Happiness
    if set_dict['Happiness'] != 255:
        text += f"  \nHappiness: {int(set_dict['Happiness'])}"

    # EVs
    if sum(set_dict['EVs']) > 0:
        text += '  \nEVs: '
        ev_parts = []
        for n in range(6):
            if set_dict['EVs'][n] > 0:
                ev_parts.append(f"{int(set_dict['EVs'][n])} {INDEX_TO_STAT[n]}")
        text += ' / '.join(ev_parts)

    # Nature
    if set_dict['Nature']:
        text += f"  \n{set_dict['Nature']} Nature"

    # IVs
    if 31 * 6 - sum(set_dict['IVs']) > 0.5:
        shared_moves = list(set_dict.get('SharedMoves1', {}).keys()) + list(set_dict.get('SharedMoves2', {}).keys())
        has_hp = sum(1 for m in shared_moves if 'Hidden Power' in m)
        if show_ivs or has_hp == 0:
            text += '  \nIVs: '
            iv_parts = []
            for n in range(6):
                if set_dict['IVs'][n] < 31:
                    iv_parts.append(f"{int(set_dict['IVs'][n])} {INDEX_TO_STAT[n]}")
            text += ' / '.join(iv_parts)

    text += '  \n'

    # Moves
    move_list = list(set_dict['Moveset'])

    def safe_move_frequency(name, move, freq_dict):
        if name in freq_dict and move in freq_dict[name]:
            return freq_dict[name][move]
        return 0

    if sort_moves_alphabetical != 0:
        move_list.sort(reverse=(sort_moves_alphabetical < 0))
    if sort_moves_frequency != 0:
        move_list.sort(
            key=lambda k: safe_move_frequency(set_dict['Name'], k, move_frequency),
            reverse=(sort_moves_frequency < 0)
        )

    for m in move_list:
        if m in set_dict.get('SharedMoves2', {}):
            moves2 = list(set_dict['SharedMoves2'].keys())
            moves2_sorted = sorted(moves2, key=lambda k: sum(set_dict['SharedMoves2'][k].values()), reverse=True)
            move_text = ' / '.join(moves2_sorted)
        elif m in set_dict.get('SharedMoves1', {}):
            moves1 = list(set_dict['SharedMoves1'].keys())
            moves1_sorted = sorted(moves1, key=lambda k: set_dict['SharedMoves1'][k], reverse=True)
            move_text = ' / '.join(moves1_sorted)
        else:
            move_text = m
        text += f"- {move_text}  \n"

    return text
