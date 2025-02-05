import pandas as pd
import numpy as np

def detect_vectors(tree, branch):
    ret = []
    hierarchy = branch.split(".")
    for i in range(len(hierarchy)):
        subbranch = ".".join(hierarchy[:i+1])
        lenbranch = subbranch + "..length"
        if lenbranch in tree.keys():
            ret.append(subbranch)

    return ret

def idarray(ids, lens):
    return np.repeat(ids.values, lens.values)

def flatten_with_indices(array, base_indices=()):
    if isinstance(array, (list, pd.Series)) or hasattr(array, "__iter__"):  # Check if iterable
        result = []
        for i, sub_array in enumerate(array):
            result.extend(flatten_with_indices(sub_array, base_indices + (i,)))
        return result
    elif hasattr(array, "to_pylist"):  # Handle Arrow or similar arrays
        result = []
        for i, sub_array in enumerate(array.to_pylist()):
            result.extend(flatten_with_indices(sub_array, base_indices + (i,)))
        return result
    else:  # Return scalar value
        return [(base_indices, array)]

def process_arbitrary_dimension(df):
    column_labels = df.columns.tolist()  # Get the list of column labels
    all_results = []  # To store the processed DataFrames

    for column_label in column_labels:
        flattened_data = []
        for row_index, array in enumerate(df[column_label]):
            # Recursively flatten the array and store the indices
            flattened_data.extend(flatten_with_indices(array, (row_index,)))    

        # Extract indices and values
        indices, values = zip(*flattened_data)

        # Determine the number of dimensions dynamically
        max_dim = max(len(idx) for idx in indices)
        index_names = ['entry'] + [f"dim_{i}" for i in range(max_dim - 1)]

        # Create a MultiIndex DataFrame for this column
        multi_index = pd.MultiIndex.from_tuples(indices, names=index_names)
        result_df = pd.DataFrame(values, index=multi_index, columns=[column_label])

        # Add the result to the list
        all_results.append(result_df)

    # Concatenate all column results into a single DataFrame
    final_result = pd.concat(all_results, axis=1)

    return final_result

def loadbranches(tree, branches):
    vectors = []
    for i, branch in enumerate(branches):
        this_vectors = detect_vectors(tree, branch)
        if i == 0:
            vectors = this_vectors
        elif len(this_vectors) == 0:  # This case is ok since it will automatically broadcast
            pass
        # All the branches must have the same vector structure for this to work
        elif vectors != this_vectors:
            raise ValueError(
                f"Branches {branches[0]} and {branch} have different vector structures in the CAF."
            )

    lengths = [tree.arrays([v + "..length"], library="pd") for v in vectors]  

    for i in range(len(lengths)):
        lengths[i] = process_arbitrary_dimension(lengths[i])
    
    data = tree.arrays(branches, library="pd")
    
    # If there's no vectors, we can just return the top guy
    if len(lengths) == 0:
        data.index.name = "entry"
        df = data
    else:
        tomerge = lengths + [data]
        # Otherwise, iteratively merge the branches
        df = tomerge[0]
        df.index.name = "entry"

        # handle the rest
        for i in range(1, len(tomerge)):
            thismerge = tomerge[i]
            v_ind = i - 1

            column_labels = thismerge.columns.tolist()
            if(not (len(column_labels) == 1 and "..length" in column_labels[0])):
                thismerge =  process_arbitrary_dimension(thismerge)

            # Build the information in the right-hand table needed to do the join
            # The "upidx" will be matched to the index vector-by-vector
            for i in range(v_ind):
                thismerge[vectors[v_ind] + "..upidx" + str(i)] = idarray(df[vectors[i]+ "..index"], df[vectors[v_ind] + "..length"])

            # Inner join! Throw away rows in the right-hand with no match in the left-hand
            df = pd.merge(df, thismerge, how="inner",
                         left_on = ["entry"] + [v+"..index" for v in vectors[:v_ind]],
                         right_on = ["entry"] + [vectors[v_ind] + "..upidx" + str(i) for i in range(v_ind)],
                         validate="one_to_many")

            # Make sure no rows in the right-hand were dropped
            assert(df.shape[0] == thismerge.shape[0])

            # postprocess: build the index
            df[vectors[v_ind] + "..index"] = df.groupby(["entry"] + [v+"..index" for v in vectors[:v_ind]]).cumcount()

        # Set the index
        df.set_index([v+"..index" for v in vectors], append=True, verify_integrity=True, inplace=True)

        # Drop all the metadata info we don't need anymore
        df = df[branches]

    # Setup branch names so df reflects structure of CAF file
    bsplit = [b.split(".") for b in branches]
    # Replace any reserved names
    def unreserve(s):
        if s == "index":
            return "idx"
        if s[0].isdigit(): # make the name a legal field 
            return "I" + s
        return s

    bsplit = [[unreserve(s) for s in b] for b in bsplit]

    depth = max([len(b) for b in bsplit])

    def pad(b):
        return tuple(b + [""]*(depth - len(b)))

    df.columns = pd.MultiIndex.from_tuples([pad(b) for b in bsplit])

    return df
