range = {
    "artillery" : 2,
    "ddg" : 2
}

sight = {
    "artillery" : 2,
    "ddg" : 2
}

pDetect = 1.0

ineffectiveThreshold = 0.50 # 50%

firepower_scaling = 0.5

firepower = {}



firepower["artillery"] = {
    "ddg" : 1.0,
    "artillery" : 1.5
}

firepower["ddg"] = {
    "ddg" : 1.0,
    "artillery" : 1.5
}


defensivefp = {}



defensivefp["artillery"] = {
    "ddg" : 0,
    "artillery" : 0
}

defensivefp["ddg"] = {
    "ddg" : 0,
    "artillery" : 0
}

terrain_multiplier = {}


terrain_multiplier["ddg"] = {
    "clear": 1,
    "water": 1,
    "rough": 0.5,
    "unused": 1,
    "coast": 1,
    "urban": 0.5
}

terrain_multiplier["artillery"] = {
    "clear": 1,
    "water": 1,
    "rough": 1,
    "unused": 1,
    "coast": 2,
    "urban": 1
}

