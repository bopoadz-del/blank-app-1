from pint import UnitRegistry
ureg=UnitRegistry()

def normalize_inputs(raw, expected):
    out, used, errs = {}, {}, {}
    for k, exp_u in (expected or {}).items():
        if k not in raw: errs[k]='missing'; continue
        v=raw[k]
        if isinstance(v, dict):
            val=v.get('value'); unit=v.get('unit')
        else:
            val=v; unit=None
        try:
            if unit:
                q=val*ureg.parse_units(unit)
                out[k]=float(q.to(ureg.parse_units(exp_u)).magnitude)
                used[k]=exp_u
            else:
                out[k]=float(val); used[k]=exp_u
        except Exception as e:
            errs[k]=f'incompatible with {exp_u}: {e}'
    return out, used, errs
