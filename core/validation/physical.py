def check(value: float):
    try:
        return {'ok': (value==value)}
    except Exception:
        return {'ok': False}
