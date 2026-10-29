"""Names and constants shared by the generator, the gate and the smoke test.

Imports nothing, so it is safe to import under Isaac Sim's interpreter.
"""

# The classes that get labels. Everything else in the scene renders but is
# never annotated.
TARGET_CLASSES = ("pallet", "pallet_truck", "forklift")

# Each class's colour in the semantic-segmentation masks the writer saves,
# which the gate reads back to measure how much of each box is the object.
SEG_COLOURS = {
    "pallet": (211, 210, 223, 255),
    "pallet_truck": (11, 110, 223, 255),
    "forklift": (255, 140, 0, 255),
}

# The two camera regimes. Each name is also the camera's name in Replicator,
# its render product and its output directory.
REGIMES = ("elevated_fixed", "ground_handheld")

RESOLUTION = (960, 540)

# Relative to the Isaac assets root.
WAREHOUSE_USD = "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
