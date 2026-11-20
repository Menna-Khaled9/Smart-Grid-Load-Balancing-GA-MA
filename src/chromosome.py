
import numpy as np

from src.config import SLOT_MINUTES, SLOTS_PER_DAY


# ---------------------------------------------------------
# Build Gene Metadata
# ---------------------------------------------------------

def build_gene_metadata(appliances):

    gene_appliance_indices = []   #each gene is corresponded to which appliance
    gene_segment_slots = []
    gene_lower_bounds = []    #least available time to start gene
    gene_upper_bounds = []     #largest available time to start gene

    for appliance_index, row in appliances.iterrows():

        segment_slots = int( row["segment_minutes"] // SLOT_MINUTES )

        duration_slots = int(row["duration_minutes"] // SLOT_MINUTES )

        if segment_slots <= 0:
            raise ValueError(
                f"Invalid segment length for {row['appliance_id']}"
            )

        number_of_blocks = duration_slots // segment_slots

        earliest_slot = int( row["earliest_minute"] // SLOT_MINUTES )

        upper_slot = int(
            (
                row["latest_minute"]
                + row["duration_minutes"]
                - row["segment_minutes"]
            ) // SLOT_MINUTES
        )

        for _ in range(number_of_blocks):

            gene_appliance_indices.append(appliance_index)

            gene_segment_slots.append(segment_slots)

            gene_lower_bounds.append(earliest_slot)

            gene_upper_bounds.append(upper_slot)

    metadata = {

        "gene_appliance_indices": np.asarray(
            gene_appliance_indices,
            dtype=int
        ),

        "gene_segment_slots": np.asarray(
            gene_segment_slots,
            dtype=int
        ),

        "gene_lower_bounds": np.asarray(
            gene_lower_bounds,
            dtype=int
        ),

        "gene_upper_bounds": np.asarray(
            gene_upper_bounds,
            dtype=int
        ),
    }

    return metadata


# ---------------------------------------------------------
# Build Original Chromosome
# ---------------------------------------------------------

def build_original_chromosome(appliances):

    genes = []

    for _, row in appliances.iterrows():

        segment_slots = int( row["segment_minutes"] // SLOT_MINUTES )

        duration_slots = int( row["duration_minutes"] // SLOT_MINUTES)

        number_of_blocks = duration_slots // segment_slots

        preferred_slot = int( row["preferred_minute"] // SLOT_MINUTES )

        genes.extend( 
            preferred_slot + segment_slots * block_index
            for block_index in range(number_of_blocks)  )   # example :[ 8 , 12 ]  first segment starts in slot 8
                                                            #second slot segment starts in slot 12
    return np.asarray(genes, dtype=int)


# ---------------------------------------------------------
# Build Coverage Matrix
# ---------------------------------------------------------

def build_coverage_matrix(genes, appliances, metadata):

    genes = np.asarray(genes, dtype=int)

    gene_app = metadata["gene_appliance_indices"]

    segment_slots = metadata["gene_segment_slots"]

    if len(genes) != len(gene_app):
        raise ValueError(
            f"Expected {len(gene_app)} genes, "
            f"but received {len(genes)}."
        )

    coverage = np.zeros(
        (len(appliances), SLOTS_PER_DAY),
        dtype=int
    )

    for start, app_index, block_length in zip(
        genes,
        gene_app,
        segment_slots
    ):

        # Clip only to avoid indexing outside the daily matrix.
        clipped_start = max(int(start), 0)

        clipped_end = min(
            int(start + block_length),
            SLOTS_PER_DAY
        )

        if clipped_start < clipped_end:

            coverage[
                app_index,
                clipped_start:clipped_end
            ] += 1

    return coverage
