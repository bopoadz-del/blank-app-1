from prometheus_client import Counter, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
registry=CollectorRegistry(auto_describe=True)
runs_total=Counter('runs_total','Total runs submitted',registry=registry)
decisions_total=Counter('decisions_total','Decisions made',['decision'],registry=registry)

def expose_metrics():
    return generate_latest(registry), CONTENT_TYPE_LATEST
