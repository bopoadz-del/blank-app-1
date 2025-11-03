def decide(confidence: float, auto: float, reco: float):
    if confidence>=auto: return 'AUTO'
    if confidence>=reco: return 'RECOMMEND'
    return 'REVIEW'
