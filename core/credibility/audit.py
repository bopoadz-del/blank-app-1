import hashlib, json

def checksum_for(code, expression, inputs_schema, version):
    blob=json.dumps({'code':code,'expr':expression,'inputs':inputs_schema,'ver':version}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()

def lineage(code, version, checksum, normalized_inputs, stages):
    return {'formula_code':code,'formula_version':version,'checksum':checksum,'normalized_inputs':normalized_inputs,'stages':stages}
