AGT FALL 2022 planning

Framework:

- Given a list of seg ids
- Given a tool to find the next seg ID ; Returns: A DF of seg_ids and confidence that they are the next ones
- Approach in a depth first manner
- Use a queue
- Push first seg_id, run the algorithm to get the next ones, and push the next most likely and so on and so forth
- Run while – confidence is greater than minimum threshold for extension || reach a seg_id with a soma

For orphan in:

- Test how the current tool is working with orphan extensions

Pseudocoding the framework:

while(confidence > min_threshold && next_seg_id.num_soma==(0))
If queue is empty:
add next seg id from seg_ids_to_extend

    Pop next_seg_id from queue

    If next_seg_id is in seg_ids_to_extend array (seg_ids_to_extend is a master array that's given)
        Remove from seg_ids_to_extend

    Run algorithm to retrieve extensions for next_seg_id(given algorithm)

    #Update next_seg_id
    next_seg_id = extension with highest confidence next_seg_id

    Remove next_seg_id from the master seg_ids_to_extend array

    Push next_seg_id onto queue


tip_finding/tip_finding.py/endpoints_from_rid() - to find tips

pass endpoints and root id into drive/segment_points()