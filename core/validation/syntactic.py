def check(inputs: dict, required: list):
    missing=[k for k in required if k not in inputs]
    return {'ok': len(missing)==0, 'missing': missing}
