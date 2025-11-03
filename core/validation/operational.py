def enforce(value: float):
    return {'ok': True, 'rounded': round(value,6), 'text': str(round(value,6))}
