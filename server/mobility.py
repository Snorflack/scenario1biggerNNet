import math

Infinity = math.inf

cost = {}
    

cost["ddg"] = {
    "clear": 100,
    "water": 50,
    "rough": Infinity,
    "unused": Infinity,
    "coast": 100,
    "urban": 100
    
}

cost["artillery"] = {
    "clear": Infinity,
    "water": Infinity,
    "rough": Infinity,
    "unused": Infinity,
    "coast": Infinity,
    "urban": Infinity
}

stackingLimit = 1
