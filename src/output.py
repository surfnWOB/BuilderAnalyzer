"""
Output module for BuilderAnalyzer.
Handles writing statistics and builder files.
"""

import os
from typing import List, Dict, Optional, Any
import logging

from .config import Config
from .set_parser import print_set

logger = logging.getLogger(__name__)

# Output directory structure
OUTPUT_DIR = 'output'


class OutputWriter:
    """Writes analysis results to files."""

    def __init__(self, config: Config, output_template: str):
        """
        Initialize the output writer.

        Args:
            config: Configuration object
            output_template: Base name for output files
        """
        self.config = config
        self.template = output_template

    def _get_output_path(self, gen: str, subdir: str, filename: str) -> str:
        """
        Get the full output path for a file, creating directories as needed.

        Args:
            gen: Generation string (e.g., 'gen3ou') or 'combined' for combined files
            subdir: Subdirectory type ('statistics', 'sets', 'builders', 'plots', 'archetypes')
            filename: The filename to write

        Returns:
            Full path to the output file
        """
        dir_path = os.path.join(OUTPUT_DIR, gen, subdir)
        os.makedirs(dir_path, exist_ok=True)
        return os.path.join(dir_path, filename)

    def write_statistics_txt(self, gen: str, core_list: List[dict], mpmi_list: List[dict],
                             lead_list: dict, team_list: List[dict], team_preview: bool,
                             analyze_teams: bool, mon_frequency: dict) -> None:
        """Write usage and synergy statistics to text files."""
        logger.info(f"Writing statistics for {gen}...")

        max_name_len = max((len(s) for s in mon_frequency.keys()), default=18)
        total_mons = sum(mon_frequency.values())
        num_teams = len(team_list)

        for stat_type in ['usage', 'synergy']:
            filename = self._get_output_path(gen, 'statistics', f"{stat_type}_statistics.txt")
            with open(filename, 'w', encoding='utf-8', errors='ignore') as f:
                if analyze_teams:
                    if not team_preview and lead_list:
                        self._write_lead_stats(f, lead_list, num_teams, max_name_len)

                    for p in range(self.config.max_core_num):
                        if stat_type == 'usage':
                            sorted_cores = sorted(
                                core_list[p].items(),
                                key=lambda x: x[1],
                                reverse=True
                            )
                        else:
                            sorted_cores = sorted(
                                core_list[p].items(),
                                key=lambda x: (x[1] ** self.config.usage_weight[p]) * mpmi_list[p].get(x[0], 0),
                                reverse=True
                            )

                        self._write_core_stats(f, sorted_cores, p, mpmi_list[p], num_teams, max_name_len)
                else:
                    self._write_mon_frequency(f, mon_frequency, total_mons)

        logger.info(f"Statistics written for {gen}")

    def write_statistics_csv(self, gen: str, core_list: List[dict], mpmi_list: List[dict],
                             lead_list: dict, team_list: List[dict], team_preview: bool,
                             analyze_teams: bool, mon_frequency: dict) -> None:
        """Write usage and synergy statistics to CSV files."""
        num_teams = len(team_list)
        total_mons = sum(mon_frequency.values())

        for stat_type in ['usage', 'synergy']:
            filename = self._get_output_path(gen, 'statistics', f"{stat_type}_statistics.csv")
            with open(filename, 'w', encoding='utf-8', errors='ignore') as f:
                if analyze_teams:
                    if not team_preview and lead_list:
                        f.write("Lead Stats\n")
                        f.write("Counts,Freq (%),Lead\n")
                        for lead, freq in sorted(lead_list.items(), key=lambda x: x[1], reverse=True):
                            f.write(f"{freq},{freq/num_teams*100:.3f},{lead}\n")
                        f.write("\n")

                    for p in range(self.config.max_core_num):
                        if stat_type == 'usage':
                            sorted_cores = sorted(core_list[p].items(), key=lambda x: x[1], reverse=True)
                        else:
                            sorted_cores = sorted(
                                core_list[p].items(),
                                key=lambda x: ((x[1] + 0.001) ** self.config.usage_weight[p]) * mpmi_list[p].get(x[0], 0),
                                reverse=True
                            )

                        header = "Counts,Freq (%)" + (",Synergy" if p > 0 else "") + ",Pokemon\n"
                        f.write(f"{p+1}-Cores\n{header}")

                        for core, freq in sorted_cores:
                            if freq == 0 and not self.config.show_missing_mon_cores:
                                continue
                            synergy = f",{mpmi_list[p].get(core, 0):.2f}" if p > 0 else ""
                            core_names = ",".join(core)
                            f.write(f"{freq},{freq/num_teams*100:.3f}{synergy},{core_names}\n")
                        f.write("\n")
                else:
                    f.write("Pokemon Frequency\n")
                    f.write("Counts,Freq (%),Pokemon\n")
                    for mon, freq in sorted(mon_frequency.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"{freq},{freq/total_mons*100:.3f},{mon}\n")

    def _write_lead_stats(self, f, lead_list: dict, num_teams: int, max_name_len: int) -> None:
        """Write lead statistics section."""
        f.write("Team Lead Arranged by Frequency\n")
        f.write(" Counts | Freq (%) | Lead\n")
        for lead, freq in sorted(lead_list.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{freq:7d} | {freq/num_teams*100:8.3f} | {lead:>{max_name_len}}\n")
        f.write("\n")

    def _write_core_stats(self, f, sorted_cores: List, p: int, mpmi: dict,
                          num_teams: int, max_name_len: int) -> None:
        """Write core statistics section."""
        if p == 0:
            f.write("Pokemon Arranged by Frequency\n")
            f.write(" Counts | Freq (%) | Pokemon\n")
        else:
            f.write(f"{p+1}-Cores Arranged by Frequency\n")
            f.write(" Counts | Freq (%) | Synergy | Cores\n")

        for core, freq in sorted_cores:
            if freq == 0 and not self.config.show_missing_mon_cores:
                continue
            f.write(f"{freq:7d} | {freq/num_teams*100:8.3f} | ")
            if p > 0:
                f.write(f"{mpmi.get(core, 0):7.2f} | ")
            f.write(", ".join(f"{c:>{max_name_len}}" for c in core))
            f.write("\n")
        f.write("\n")

    def _write_mon_frequency(self, f, mon_frequency: dict, total: int) -> None:
        """Write Pokemon frequency section."""
        f.write("Pokemon Arranged by Frequency\n")
        f.write(" Counts | Freq (%) | Pokemon\n")
        for mon, freq in sorted(mon_frequency.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{freq:7d} | {freq/total*100:8.3f} | {mon}\n")

    def write_sets_file(self, gen: str, set_list: List[dict], mon_frequency: dict,
                        move_frequency: dict, frac_threshold: float) -> None:
        """Write sets compendium file."""
        if frac_threshold == 0:
            sets_filename = "sets.txt"
        else:
            sets_filename = f"sets_cut_{int(1/frac_threshold)}.txt"

        filename = self._get_output_path(gen, 'sets', sets_filename)
        logger.info(f"Writing sets to {filename}...")

        with open(filename, 'w', encoding='utf-8', errors='ignore') as f:
            self._write_sets_header(f, frac_threshold)

            for s in set_list:
                mon_freq = mon_frequency.get(s['Name'], 1)
                frac = s.get('CountMoves', 1) / mon_freq if mon_freq > 0 else 0

                if frac >= frac_threshold:
                    f.write(print_set(
                        s, move_frequency,
                        self.config.show_shiny,
                        self.config.show_ivs,
                        self.config.show_nicknames,
                        self.config.sort_moves_by_alphabetical,
                        self.config.sort_moves_by_frequency
                    ))

                    if self.config.show_statistics_in_sets:
                        f.write('-' * 28 + '\n')
                        self._write_set_stats(f, s, mon_freq)

                    f.write(f"Total: {s.get('CountMoves', 1)} | {frac*100:.1f}%\n\n")

    def _write_sets_header(self, f, frac_threshold: float) -> None:
        """Write header for sets file."""
        f.write(f"Built from {self.template}.txt\n")
        f.write('-' * 50 + '\n')
        f.write("Fraction of sets ignored: ")
        if frac_threshold == 0:
            f.write("None")
        else:
            f.write(f"{frac_threshold*100:.2f}% or 1/{int(1/frac_threshold)}")
        f.write('\n')
        f.write(f"EV movements ignored: {self.config.ev_threshold}\n")
        f.write("IV sum deviation from 31 ignored: ")
        f.write(str(self.config.iv_threshold) if self.config.iv_threshold < 31*6 else "All")
        f.write('\n')
        f.write(f"Moveslots combined: {self.config.combine_moves}\n")
        f.write("Move Sort Order: ")
        if self.config.sort_moves_by_frequency == 1:
            f.write("Increasing Frequency")
        elif self.config.sort_moves_by_frequency == -1:
            f.write("Decreasing Frequency")
        elif self.config.sort_moves_by_alphabetical == 1:
            f.write("Increasing Alphabetical")
        elif self.config.sort_moves_by_alphabetical == -1:
            f.write("Decreasing Alphabetical")
        else:
            f.write("Retained From Import")
        f.write('\n')
        f.write(f"Show IVs when Hidden Power Type is ambiguous: {'Yes' if self.config.show_ivs else 'No'}\n")
        f.write(f"Show Shiny: {'Yes' if self.config.show_shiny else 'No'}\n")
        f.write(f"Show Nicknames: {'Yes' if self.config.show_nicknames else 'No'}\n")
        f.write('-' * 50 + '\n')
        f.write('To read statistics:\nCounts | Frequency given the same Pokemon (%)\n\n')

    def _write_set_stats(self, f, s: dict, mon_freq: int) -> None:
        """Write detailed statistics for a set."""
        shared1 = s.get('SharedMoves1', {})
        shared2 = s.get('SharedMoves2', {})

        if shared1 or shared2:
            move_combos = []
            if shared2:
                for m2 in shared2:
                    for m1 in shared2[m2]:
                        move_combos.append((f"{m1} / {m2}", shared2[m2][m1]))
            elif shared1:
                for m in shared1:
                    move_combos.append((m, shared1[m]))

            move_combos.sort(key=lambda x: x[1], reverse=True)
            max_name_len = max(len(m[0]) for m in move_combos)
            max_count_len = max(len(str(m[1])) for m in move_combos)

            for name, count in move_combos:
                pct = count / mon_freq * 100 if mon_freq > 0 else 0
                f.write(f"{name:<{max_name_len}}: {count:>{max_count_len}} | {pct:.1f}%\n")

    def write_sorted_builder(self, gen: str, team_list: List[dict], set_list: List[dict],
                             move_frequency: dict, archetype_analyzer: Any = None) -> None:
        """Write sorted builder file for a generation."""
        filename = self._get_output_path(gen, 'builders', 'sorted_builder.txt')
        logger.info(f"Writing sorted builder to {filename}...")

        with open(filename, 'w', encoding='utf-8', errors='ignore') as f:
            for team in team_list:
                if team['Anomalies'] > self.config.anomaly_threshold:
                    continue

                f.write(f"=== [{gen}] {team['Folder']}")

                if self.config.print_archetype_label and archetype_analyzer:
                    # Note: Archetype finding would need category mapping
                    pass

                f.write(f"{team['Name']} ===\n\n")

                for i in range(team['Index'][0], team['Index'][1]):
                    f.write(print_set(
                        set_list[i], move_frequency,
                        True, True, True,
                        self.config.sort_moves_by_alphabetical,
                        self.config.sort_moves_by_frequency
                    ))
                    f.write('\n')
                f.write('\n')

    def write_incomplete_teams(self, team_list: List[dict], set_list: List[dict],
                               move_frequency: dict) -> None:
        """Write incomplete teams to file."""
        filename = self._get_output_path('combined', 'builders', 'incomplete.txt')
        logger.info(f"Writing incomplete teams to {filename}...")

        sorted_teams = sorted(team_list, key=lambda x: x['Line'])

        with open(filename, 'w', encoding='utf-8', errors='ignore') as f:
            for team in sorted_teams:
                f.write(f"=== [{team['Gen']}] {team['Name']} ===\n\n")
                for i in range(team['Index'][0], team['Index'][1]):
                    f.write(print_set(set_list[i], move_frequency, True, True, True, 0, 0))
                    f.write('\n')
                f.write('\n')

    def write_full_builder(self, generations: List[str], include_incomplete: bool) -> None:
        """Combine all sorted builders into a single file."""
        filename = self._get_output_path('combined', 'builders', 'full_sorted_builder.txt')
        logger.info(f"Writing full sorted builder to {filename}...")

        with open(filename, 'w', encoding='utf-8', errors='ignore') as fo:
            if include_incomplete:
                try:
                    incomplete_path = self._get_output_path('combined', 'builders', 'incomplete.txt')
                    with open(incomplete_path, encoding='utf-8', errors='ignore') as fi:
                        fo.write(fi.read())
                except FileNotFoundError:
                    logger.warning("Incomplete teams file not found")

            for gen in generations:
                try:
                    gen_builder_path = self._get_output_path(gen, 'builders', 'sorted_builder.txt')
                    with open(gen_builder_path, encoding='utf-8', errors='ignore') as fi:
                        fo.write(fi.read())
                except FileNotFoundError:
                    logger.warning(f"Sorted builder for {gen} not found")

    def write_archetype_statistics(self, gen: str, archetype_analyzer: Any,
                                   cat_core_list: List[dict], category_nics: dict,
                                   team_list: List[dict]) -> None:
        """Write archetype analysis results to file."""
        if archetype_analyzer is None or archetype_analyzer.p_mat is None:
            logger.warning("No archetype data to write")
            return

        filename = self._get_output_path(gen, 'archetypes', 'archetype_statistics.txt')
        logger.info(f"Writing archetype statistics to {filename}...")

        p_mat = archetype_analyzer.p_mat
        cat_indiv_list = archetype_analyzer.cat_indiv_list
        archetype_order = archetype_analyzer.archetype_order
        num_archetypes = archetype_analyzer.actual_num_archetypes
        num_teams = len(team_list)

        max_name_len = max((len(category_nics.get(c, str(c))) for c in cat_indiv_list), default=18)

        with open(filename, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(f"Built from {self.template}.txt\n")
            f.write('-' * 50 + '\n')
            f.write(f"Number of archetypes: {num_archetypes}\n")
            f.write(f"Exponent: {self.config.exponent:.3f}\n")
            f.write(f"Gamma spectral: {self.config.gamma_spectral:.3f}\n")
            f.write('-' * 50 + '\n\n')

            for ii in archetype_order:
                f.write(f"Archetype {archetype_order.index(ii) + 1}\n")
                f.write(" Counts | Freq (%) | Confidence | Pokemon\n")

                cat_sorted = sorted(
                    range(len(cat_indiv_list)),
                    key=lambda x: p_mat[ii, x] * (cat_core_list[0][(cat_indiv_list[x],)] ** self.config.gamma_archetypes),
                    reverse=True
                )

                for cat_idx in cat_sorted:
                    cat = cat_indiv_list[cat_idx]
                    freq = cat_core_list[0][(cat,)]
                    if freq == 0:
                        continue

                    cat_name = category_nics.get(cat, str(cat))
                    conf = p_mat[ii, cat_idx]
                    f.write(f"{freq:7d} | {freq/num_teams*100:8.3f} | {conf:10.2f} | {cat_name:>{max_name_len}}\n")
                f.write('\n')
