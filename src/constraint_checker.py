
import numpy as np

from src.config import SLOT_MINUTES, SLOTS_PER_DAY



############ Check All Hard Constraints

def check_constraints(genes, appliances, metadata, coverage):

    genes = np.asarray(genes, dtype=int)

    gene_segment_slots = metadata["gene_segment_slots"]

    lower = metadata["gene_lower_bounds"]

    upper = metadata["gene_upper_bounds"]

    violations = 0

    # -----------------------------------------------------
    # 1. Allowed Start Time Constraint
    # -----------------------------------------------------

    for gene_index, start in enumerate(genes):

        if ( start < lower[gene_index] or start > upper[gene_index]):     #validate that appliance mustn't work be of the allowed lower and upper bounds
            violations += 1

    # -----------------------------------------------------
    # 2. Daily Time Boundary Constraint
    # -----------------------------------------------------

    for start, block_length in zip( genes, gene_segment_slots ):

        if ( start < 0 or start + block_length > SLOTS_PER_DAY ):   # #validate that it doesn't work before day and after the day ends

            violations += 1

    # Convert coverage to ON/OFF
    appliance_on = coverage > 0


   
   
    ############ 3. No Overlapping Segments Constraint

     
#Check Overlapping Segments (after we specify time for each appliance , must check that the appliance works 2 times in same segment due to segment overlapping)
  #if appliance A01 has segment 1  ====>  slots :  10 , 11 ,12  , segment 2 ====> slots :  12 , 13 , 14 
    
    for app_index in range(len(appliances)):

        if np.any(coverage[app_index] > 1):

            violations += 1

    # -----------------------------------------------------
    ######### 4. Required Operating Duration Constraint


     #VALIDATE THAT EACH APPLIANCE WORKS ON THE REQUIRED SLOTS onlyyyyyyyyyy
    for app_index, row in appliances.iterrows():

        required_slots = int(
            row["duration_minutes"] // SLOT_MINUTES
        )

        actual_slots = int(
            appliance_on[app_index].sum()
        )

        if actual_slots != required_slots:

            violations += 1

    # -----------------------------------------------------
    ########### 5. Non-Interruptible Appliance Constraint
 
     # #APPLIANCE MUSTN'T BE INTERRUPTED
    for app_index, row in appliances.iterrows():

        if not bool(row["interruptible"]):

            occupied = np.flatnonzero(
                appliance_on[app_index]
            )

            if len(occupied) > 0:

                is_contiguous = (
                    occupied[-1] - occupied[0] + 1
                    == len(occupied)
                )

                if not is_contiguous:

                    violations += 1

    return int(violations)
