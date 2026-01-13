"""
Archetype analysis module for BuilderAnalyzer.
Handles spectral clustering for team archetype identification.
"""

from typing import List, Dict, Tuple, Optional
import math
import logging

logger = logging.getLogger(__name__)

# Optional imports for archetype analysis
try:
    import numpy as np
    import skfuzzy as fuzz
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend to prevent blocking
    import matplotlib.pyplot as plt
    ARCHETYPE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Archetype analysis dependencies not available: {e}")
    ARCHETYPE_AVAILABLE = False


class ArchetypeAnalyzer:
    """Performs spectral clustering for archetype identification."""

    def __init__(self, config):
        """
        Initialize the archetype analyzer.

        Args:
            config: Configuration object with archetype parameters
        """
        self.config = config
        self.cat_indiv_list = None
        self.p_mat = None
        self.vk = None
        self.cntr = None
        self.archetype_order = None
        self.actual_num_archetypes = 0  # Set during analyze()

        if not ARCHETYPE_AVAILABLE:
            logger.error("Archetype analysis requires numpy, scikit-fuzzy, and matplotlib")

    def analyze(self, cat_core_list: List[dict], show_plot: bool = False,
                save_plot: Optional[str] = None) -> Optional[Tuple]:
        """
        Perform spectral clustering to identify archetypes.

        Args:
            cat_core_list: List of category core dictionaries
            show_plot: Whether to display the FPC plot (non-blocking)
            save_plot: Path to save the plot image

        Returns:
            Tuple of (p_mat, cat_indiv_list, archetype_order) or None if unavailable
        """
        if not ARCHETYPE_AVAILABLE:
            logger.error("Cannot perform archetype analysis - dependencies missing")
            return None

        logger.info("Starting archetype analysis...")

        # Get individual categories
        self.cat_indiv_list = [c[0] for c in cat_core_list[0].keys()]
        num_cats = len(self.cat_indiv_list)
        logger.info(f"Analyzing {num_cats} categories")

        if num_cats < 3:
            logger.warning("Not enough categories for archetype analysis")
            return None

        # Build adjacency matrix
        W = np.zeros((num_cats, num_cats))
        for ii in range(num_cats):
            for jj in range(num_cats):
                core = tuple(sorted(
                    [self.cat_indiv_list[ii], self.cat_indiv_list[jj]],
                    key=lambda x: x[0]
                ))
                if core in cat_core_list[1]:
                    W[ii, jj] = cat_core_list[1][core] ** self.config.gamma_spectral

        # Compute Laplacian matrices
        D = np.diag(np.sum(W, axis=1))
        D_inv = np.diag(1.0 / np.sum(W, axis=1))
        D_inv_root = np.diag(1.0 / np.sqrt(np.sum(W, axis=1)))

        L = D - W  # Unnormalized Laplacian
        L_sym = D_inv_root @ L @ D_inv_root  # Symmetric normalized

        # Eigendecomposition
        logger.info("Computing eigendecomposition...")
        eigenval, v = np.linalg.eigh(L_sym)

        # Find optimal number of clusters using FPC
        fpcs = []
        max_clusters = min(self.config.num_archetypes_range, num_cats)
        logger.info(f"Testing cluster counts from 2 to {max_clusters}...")

        for n_centers in range(2, max_clusters):
            vk = np.zeros((num_cats, num_cats))
            for ii in range(num_cats):
                norm = math.sqrt(sum(v[ii, 0:n_centers] ** 2))
                if norm > 0:
                    vk[ii, 0:n_centers] = v[ii, 0:n_centers] / norm

            try:
                _, _, _, _, _, _, fpc = fuzz.cluster.cmeans(
                    vk[:, 0:n_centers].transpose(),
                    n_centers,
                    self.config.exponent,
                    error=0.005,
                    maxiter=1000,
                    init=None
                )
                fpcs.append(fpc)
                logger.debug(f"  {n_centers} clusters: FPC = {fpc:.4f}")
            except Exception as e:
                logger.warning(f"  {n_centers} clusters: Failed - {e}")
                fpcs.append(0)

        # Generate FPC plot
        if show_plot or save_plot:
            self._generate_fpc_plot(fpcs, show_plot, save_plot or "")

        # Determine optimal number of archetypes
        if getattr(self.config, 'auto_num_archetypes', False):
            optimal = self._find_optimal_archetypes(fpcs)
            self.actual_num_archetypes = min(optimal, num_cats)
            logger.info(f"Auto-selected {self.actual_num_archetypes} archetypes based on FPC analysis")
        else:
            # Use configured number, but cap at available categories
            self.actual_num_archetypes = min(self.config.num_archetypes, num_cats)
            if self.actual_num_archetypes < self.config.num_archetypes:
                logger.warning(f"Only {num_cats} categories available, using {self.actual_num_archetypes} archetypes instead of {self.config.num_archetypes}")
        logger.info(f"Final clustering with {self.actual_num_archetypes} archetypes...")

        self.vk = np.zeros((num_cats, self.actual_num_archetypes))
        for ii in range(num_cats):
            norm = math.sqrt(sum(v[ii, 0:self.actual_num_archetypes] ** 2))
            if norm > 0:
                self.vk[ii, :] = v[ii, 0:self.actual_num_archetypes] / norm

        self.cntr, self.p_mat, _, _, _, _, fpc = fuzz.cluster.cmeans(
            self.vk.transpose(),
            self.actual_num_archetypes,
            self.config.exponent,
            error=0.005,
            maxiter=1000,
            init=None
        )
        logger.info(f"Final FPC: {fpc:.4f}")

        # Determine archetype ordering by strength
        cat_count = [
            (cat_core_list[0][(self.cat_indiv_list[x],)]) ** self.config.gamma_archetypes
            for x in range(np.shape(self.p_mat)[1])
        ]
        cat_count_arr = np.tile(np.array(cat_count), (self.actual_num_archetypes, 1))
        archetype_strength = np.sum(self.p_mat * cat_count_arr, axis=1)
        self.archetype_order = sorted(
            range(self.actual_num_archetypes),
            key=lambda x: archetype_strength[x],
            reverse=True
        )

        logger.info("Archetype analysis complete")
        return self.p_mat, self.cat_indiv_list, self.archetype_order

    def _generate_fpc_plot(self, fpcs: List[float], show: bool, save_path: str) -> None:
        """Generate the FPC vs number of clusters plot."""
        fig, ax = plt.subplots()
        x_values = list(range(2, 2 + len(fpcs)))
        ax.plot(x_values, fpcs)
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("Fuzzy partition coefficient")
        ax.set_title("Archetype Clustering Analysis")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"FPC plot saved to {save_path}")

        if show:
            plt.show(block=False)
            plt.pause(0.1)

        plt.close(fig)

    def _find_optimal_archetypes(self, fpcs: List[float]) -> int:
        """
        Automatically determine the optimal number of archetypes based on FPC values.

        Uses cliff detection: find where FPC drops dramatically and select
        the number of clusters just before that cliff.

        Args:
            fpcs: List of FPC values for cluster counts 2, 3, 4, ...

        Returns:
            Optimal number of clusters (archetypes)
        """
        if not fpcs:
            return 2

        if len(fpcs) < 2:
            return 2

        min_fpc = getattr(self.config, 'min_fpc_threshold', 0.5)
        cliff_threshold = getattr(self.config, 'fpc_cliff_threshold', 0.3)

        # fpcs[0] corresponds to 2 clusters, fpcs[1] to 3 clusters, etc.
        max_fpc = fpcs[0]

        # Method 1: Find the cliff - where FPC drops dramatically
        for i in range(len(fpcs) - 1):
            if fpcs[i] > 0:
                drop_pct = (fpcs[i] - fpcs[i + 1]) / fpcs[i]
                if drop_pct > cliff_threshold:
                    # Found cliff, return k just before it
                    optimal_clusters = i + 2  # +2 because fpcs[0] = 2 clusters
                    logger.info(f"FPC analysis: max={max_fpc:.4f} at 2 clusters, "
                               f"cliff detected at {i + 3} clusters ({drop_pct:.1%} drop), "
                               f"selected {optimal_clusters} clusters (FPC={fpcs[i]:.4f})")
                    return optimal_clusters

        # Method 2: No cliff found, use minimum FPC threshold
        for i in range(len(fpcs)):
            if fpcs[i] < min_fpc:
                optimal_clusters = max(i + 1, 2)  # one before dropping below threshold
                logger.info(f"FPC analysis: max={max_fpc:.4f} at 2 clusters, "
                           f"no cliff, FPC dropped below {min_fpc} at {i + 2} clusters, "
                           f"selected {optimal_clusters} clusters (FPC={fpcs[i-1] if i > 0 else fpcs[0]:.4f})")
                return optimal_clusters

        # Method 3: FPC stays high throughout, return max tested
        optimal_clusters = len(fpcs) + 1
        logger.info(f"FPC analysis: max={max_fpc:.4f} at 2 clusters, "
                   f"FPC stayed above {min_fpc}, selected {optimal_clusters} clusters (FPC={fpcs[-1]:.4f})")
        return optimal_clusters

    def find_team_archetype(self, team: dict, cat_indiv_dict_inv: dict,
                            cat_core_list: List[dict]) -> Tuple[int, List[float]]:
        """
        Determine which archetype a team belongs to.

        Args:
            team: Team dictionary with Categories
            cat_indiv_dict_inv: Inverse mapping of categories to indices
            cat_core_list: Category core list

        Returns:
            Tuple of (archetype_index, deviation_list)
        """
        if not ARCHETYPE_AVAILABLE or self.p_mat is None or self.cntr is None or self.vk is None:
            return 0, [0] * self.actual_num_archetypes

        total_deviation = [0.0] * self.actual_num_archetypes

        for c in team.get('Categories', []):
            if c not in cat_indiv_dict_inv:
                # Handle categories not in the analysis
                num_cat_features = len(c)
                similar_cats = []
                weights = []

                for d in cat_indiv_dict_inv:
                    if d[0:num_cat_features] == c:
                        similar_cats.append(d)
                        weights.append(cat_core_list[0][(d,)])

                if similar_cats:
                    weight_sum = sum(weights)
                    weights = [w / weight_sum for w in weights] if weight_sum > 0 else weights

                    for n in range(self.actual_num_archetypes):
                        if self.config.metric_archetypes == 0:
                            # Distance-based
                            dist_list = [
                                np.sqrt(sum((self.cntr[:, n] - self.vk[cat_indiv_dict_inv[d], :]) ** 2))
                                for d in similar_cats
                            ]
                            avg_dist = sum(d * w for d, w in zip(dist_list, weights))
                            total_deviation[n] += avg_dist ** self.config.gamma_team_assignment
                        else:
                            # Probability-based
                            comp_prob_list = [
                                1 - self.p_mat[n, cat_indiv_dict_inv[d]]
                                for d in similar_cats
                            ]
                            avg_prob = sum(p * w for p, w in zip(comp_prob_list, weights))
                            total_deviation[n] += avg_prob ** self.config.gamma_team_assignment
            else:
                for n in range(self.actual_num_archetypes):
                    if self.config.metric_archetypes == 0:
                        dist = np.sqrt(sum(
                            (self.cntr[:, n] - self.vk[cat_indiv_dict_inv[c], :]) ** 2
                        ))
                        total_deviation[n] += dist ** self.config.gamma_team_assignment
                    else:
                        total_deviation[n] += (
                            1 - self.p_mat[n, cat_indiv_dict_inv[c]] ** self.config.gamma_team_assignment
                        )

        best_archetype = total_deviation.index(min(total_deviation))
        return best_archetype, total_deviation

    def get_category_archetype_membership(self, cat_index: int, archetype_index: int) -> float:
        """Get the membership probability of a category in an archetype."""
        if self.p_mat is None:
            return 0.0
        return self.p_mat[archetype_index, cat_index]
