import rich

def rich_print(*args, **kwargs):
    """Prints stuff using rich, with a fallback to standard print."""
    try:
        rich.print(*args, **kwargs)
    except Exception:
        print(*args, **kwargs)
