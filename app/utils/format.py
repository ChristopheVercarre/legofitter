"""Small display helpers shared across the project.

Nothing here touches the filesystem or the model — these only turn numbers
into the strings a human reads in a log.
"""


def format_duration(seconds: float) -> str:
    """Seconds as a human-readable duration: 1h 42m 18s, 7m 05s, 43s.

    Training runs span three orders of magnitude -- a laptop smoke test takes
    seconds, a VGG16 run on the VM takes hours -- so a bare "6138.4 seconds"
    is the one format that reads badly at every scale.

    Minutes and seconds are zero-padded once a larger unit is present, so a
    column of durations lines up.
    """
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"
