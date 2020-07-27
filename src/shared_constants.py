"""
Constants shard between multiple files to stay consistent.
"""

# based on experience sensors are often index using these keys
DIMENSIONS_KEY_LIST = ["x", "y", "z", 0, 1, 2, 3, 4, 5, 6, 7, 8]
TIMESTAMP_KEY = "timestamp"
ACTION_ID_COL = "action_id"
SEGMENTATION_NO_OVERLAP = "segmentation_no_window"
SEGMENTATION_OVERLAP = "segmentation_window"
