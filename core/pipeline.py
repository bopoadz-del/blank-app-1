from sqlalchemy.orm import Session
from sqlalchemy import text
from core.context.detector import detect_context
from core.execution.sympy_executor import SafeExecutor
from core.validation import syntactic, dimensional, physical, empirical, operational
from core.credibility.gate import decide
from core.credibility.audit import checksum_for, lineage
from core.selection.selector import select_candidates
from core.credibility.review_repo import ReviewRepo
from core.tinker.tracker import TinkerTracker
import json, os
AUTO=float(os.getenv('CONFIDENCE_AUTO_THRESHOLD','0.85'))
RECO=float(os.getenv('CONFIDENCE_RECOMMEND_THRESHOLD','0.70'))

def _load_formula(db:Session, code:str):
    row=db.execute(text('SELECT code,expression,inputs,output_unit,domain,version,target FROM formulas WHERE code=:c'),{'c':code}).mappings().first()
    if not row: raise ValueError('Formula not found')
    return dict(row)

def _run_once(db:Session, code:str, raw_inputs:dict, ctx:dict):
    f=_load_formula(db, code)
    expected=f['inputs'] if isinstance(f['inputs'],dict) else {}
    v_syn=syntactic.check(raw_inputs, list(expected.keys()))
    v_dim=dimensional.check(raw_inputs, expected)
    if not v_dim['ok']: raise ValueError('Dimensional check failed')
    inputs=v_dim['normalized']
    target_symbol=(f.get('target') or f['expression'].split('-')[0].strip().split()[0])
    ex=SafeExecutor(f['expression'])
    result_value=ex.solve_for(target_symbol, inputs)
    v_phy=physical.check(result_value)
    v_emp=empirical.check(f.get('domain',''), target_symbol.lower(), result_value)
    v_op=operational.enforce(result_value)
    confidence=0.5
    if ctx.get('domain')==f.get('domain'): confidence+=0.3
    if all([v_syn['ok'], v_dim['ok'], v_phy['ok'], v_emp['ok'], v_op['ok']]): confidence+=0.2
    if confidence>1.0: confidence=1.0
    decision=decide(confidence, AUTO, RECO)
    chk=checksum_for(f['code'], f['expression'], f['inputs'], f['version'])
    stages={'syntactic':v_syn,'dimensional':{'ok':v_dim['ok'],'errors':v_dim['errors']},'physical':v_phy,'empirical':v_emp,'operational':v_op}
    lin=lineage(f['code'], f['version'], chk, inputs, stages)
    data={'context':json.dumps(ctx),'formula_code':f['code'],'inputs':json.dumps(raw_inputs),'result_value':v_op.get('rounded',result_value),'result_unit':f['output_unit'],'confidence':confidence,'decision':decision,'validation':json.dumps(stages),'lineage':json.dumps(lin)}
    rid=db.execute(text('INSERT INTO runs(context,formula_code,inputs,result_value,result_unit,confidence,decision,validation,lineage) VALUES (:context::jsonb,:formula_code,:inputs::jsonb,:result_value,:result_unit,:confidence,:decision,:validation::jsonb,:lineage::jsonb) RETURNING id'), data).mappings().first()['id']
    db.commit()
    if decision=='REVIEW': ReviewRepo(db).create_for_run(rid)
    try: TinkerTracker(db).update(f['code'], True)
    except Exception: pass
    return rid, {'decision':decision,'confidence':confidence,'result_value':data['result_value'],'result_unit':f['output_unit'],'validation':stages}

def run_pipeline(db:Session, formula_code:str, inputs:dict, context_text:str):
    ctx=detect_context(context_text)
    if formula_code:
        rid,res=_run_once(db, formula_code, inputs, ctx)
        return rid,res,formula_code
    for code in select_candidates(db, ctx.get('domain'), inputs):
        try:
            rid,res=_run_once(db, code, inputs, ctx)
            return rid,res,code
        except Exception:
            try: TinkerTracker(db).update(code, False)
            except Exception: pass
            continue
    raise ValueError('No candidate formula succeeded')
