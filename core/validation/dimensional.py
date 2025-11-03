from core.units.unit_service import normalize_inputs

def check(inputs: dict, expected_units: dict):
    normalized, used, errs = normalize_inputs(inputs, expected_units)
    return {'ok': len(errs)==0, 'errors': errs, 'normalized': normalized, 'used_units': used}
