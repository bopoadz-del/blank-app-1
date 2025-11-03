def detect_context(text:str):
    t=(text or '').lower()
    domain = 'wind' if 'wind' in t else ('storm' if 'storm' in t or 'pipe' in t else 'structural' if 'beam' in t else None)
    return {'domain':domain,'confidence':0.7 if domain else 0.4}
