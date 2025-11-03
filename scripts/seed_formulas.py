import os, csv, json, sys
from sqlalchemy import create_engine, text
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    print('DATABASE_URL not set', file=sys.stderr); sys.exit(1)
engine = create_engine(DATABASE_URL)

def upsert_formula(conn, row):
    code=row['code']; name=row['name']; expression=row['expression']
    inputs=json.loads(row['inputs']); output_unit=row['output_unit']; domain=row['domain']
    version=int(row.get('version',1)); target=row.get('target')
    conn.execute(text("""
        INSERT INTO formulas(code,name,expression,inputs,output_unit,domain,version,target)
        VALUES (:code,:name,:expression,:inputs::jsonb,:output_unit,:domain,:version,:target)
        ON CONFLICT (code) DO UPDATE SET
          name=EXCLUDED.name,
          expression=EXCLUDED.expression,
          inputs=EXCLUDED.inputs,
          output_unit=EXCLUDED.output_unit,
          domain=EXCLUDED.domain,
          version=EXCLUDED.version,
          target=EXCLUDED.target
    """), { 'code':code,'name':name,'expression':expression,'inputs':json.dumps(inputs),
             'output_unit':output_unit,'domain':domain,'version':version,'target':target })

def main():
    path=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','seeds','formulas.csv'))
    with engine.begin() as conn:
        with open(path, newline='', encoding='utf-8') as f:
            reader=csv.DictReader(f); count=0
            for row in reader:
                upsert_formula(conn,row); count+=1
    print(f'Seeded {count} formulas from {path}')

if __name__=='__main__': main()
