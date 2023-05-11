import numpy as np
import sys
import os 
from pathlib import Path
# sys.path.append("..")
parent_folder = Path(os.path.abspath(__file__)).parent.parent
sys.path.append(str(parent_folder))
import map
import unit
import math
import Game
import game
import copy
from alphazero_observation import nnetObservation 
import scenario
import json

# Alpha Star General game implementation for Atlatl's city-inf-5

# The portable game board will be portable (JSON convertible) Python objects with param and state properties
# The neural network input board format will be a pair with the phasing faction plus a numpy state
class AtlatlGame(Game.Game):

    # Allows file to be invoked from any directory and still find the scenarios
    from pathlib import Path
    path = Path(__file__)
    
    scenariodir = str(path.resolve().parents[2]) + '/server/scenarios/' #str(path.parent.parent.parent) + "/scenarios/"
    print('scenariodir',scenariodir)

    registry = {
        "Scenario1middleIsland.scn":
            {"gen":scenario.from_file_factory("Scenario1middleIsland.scn",scenariodir),
            "max_score":100,
            "all_units_move_one":False},
        "Scenario2Mainland_and_Island.scn":
            {"gen":scenario.from_file_factory("Scenario2Mainland_and_Island.scn",scenariodir),
            "max_score":271,
            "all_units_move_one":False},
        "Scenario3multiIsland.scn":
            {"gen":scenario.from_file_factory("Scenario3multiIsland.scn",scenariodir),
            "max_score":100,
            "all_units_move_one":False},
        "Scenario4landApproach.scn":
            {"gen":scenario.from_file_factory("Scenario4landApproach.scn",scenariodir),
            "max_score":121.75,
            "all_units_move_one":False},
        "Scenario5Chokepoint.scn":
            {"gen":scenario.from_file_factory("Scenario5Chokepoint.scn",scenariodir),
            "max_score":300,
            "all_units_move_one":False},
        "Scenario6TightChannel.scn":
            {"gen":scenario.from_file_factory("Scenario6TightChannel.scn",scenariodir),
            "max_score":100,
            "all_units_move_one":False}
    }

    def __init__(self, scenario_name):
        self.game = None
        print('scenario_name',scenario_name)
        self.game_spec = AtlatlGame.registry[scenario_name]
        self.scenario_generator = self.game_spec["gen"]
        # self.scenario_generator = scenario.clear_square_factory(size=5, min_units=2, max_units=4, num_cities=1, scenarioSeed=4025, scenarioCycle=1, balance=False, max_phases=10, fog_of_war=False)
        # self.scenario_generator = scenario.from_file_factory("atomic-5x5.scn")
        #self.scenario_generator = scenario.from_file_factory("2v1-5x5.scn")
        self.getInitBoard() # Initialize so board size, etc. can be determined

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network, but actually
                        this will be a portable, i.e. JSON compatible, Python object)
        """
        self.game = game.Game(self.scenario_generator()) # Atlatl
        param = self.game.parameters()
        state = self.game.initial_state()
        return {"param":param, "state":state}

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions.
        """
        dims = self.game.mapData.getDimensions()
        return (dims["width"], dims["height"])

    def getActionSize(self):
        """
        This is hardcoded at the time but the actionSize needs to be the size of the board X * Y * movements + 1 for 'Pass'
        Returns:
            actionSize: number of all possible actions. Will be based on four units per faction.
        
        if self.game_spec["all_units_move_one"]:
            return 5*5*6+1 # 5x5 grid, 6 directions, plus end turn (pass), equals 151
        else:
            return 7*7*18+1 # as above, but units can move two hexes, equals 451 Scenario2 1801  Make sure this fits the board size (7x7)  7x7x18+1 = 883
        """
        """
    Returns:
        actionSize: number of all possible actions. Will be based on the number of units per faction.
    """
        num_blue_units = 3
        num_red_units = 2
        total_units = num_blue_units + num_red_units

        move_actions_per_unit = 6*7*7 if self.game_spec["all_units_move_one"] else 18*7*7
        shoot_actions_per_unit = 12*7*7
        do_nothing_action = 1

        total_actions_per_unit = move_actions_per_unit + shoot_actions_per_unit

        return total_actions_per_unit * total_units + do_nothing_action
    
    def getNumTargetHexes(self):
        if self.game_spec["all_units_move_one"]:
            return 6
        else:
            return 18 

    evenXOffsets18 = ((0,-1), (1,-1), (1,0), (0,1), (-1,0), (-1,-1),
                        (0,-2), (1,-2), (2,-1), (2,0), (2,1), (1,1), 
                        (0,2), (-1,1), (-2,1), (-2,0), (-2,-1), (-1,-2))
    oddXOffsets18 = ((0,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0),
                        (0,-2), (1,-1), (2,-1), (2,0), (2,1), (1,2), 
                        (0,2), (-1,2), (-2,1), (-2,0), (-2,-1), (-1,-1)) 
    evenXDict = {(0,-1):0, (1,-1):1, (1,0):2, (0,1):3, (-1,0):4, (-1,-1):5,
                        (0,-2):6, (1,-2):7, (2,-1):8, (2,0):9, (2,1):10, (1,1):11, 
                        (0,2):12, (-1,1):13, (-2,1):14, (-2,0):15, (-2,-1):16, (-1,-2):17}
    oddXDict = {(0,-1):0, (1,0):1, (1,1):2, (0,1):3, (-1,1):4, (-1,0):5,
                        (0,-2):6, (1,-1):7, (2,-1):8, (2,0):9, (2,1):10, (1,2):11, 
                        (0,2):12, (-1,2):13, (-2,1):14, (-2,0):15, (-2,-1):16, (-1,-1):17}

    def targetCoordsToCode(self, mover_hex, target_hex):
        diff = (target_hex[0]-mover_hex[0], target_hex[1]-mover_hex[1])
        if mover_hex[0]%2==0:
            return AtlatlGame.evenXDict[diff]
        else:
            return AtlatlGame.oddXDict[diff]

    def codeToTargetCoords(self, start_hex, code):
        if start_hex[0]%2==0:
            delta = AtlatlGame.evenXOffsets18[code]
        else:
            delta = AtlatlGame.oddXOffsets18[code]
        return (start_hex[0]+delta[0],start_hex[1]+delta[1])

    def atlatlActionToVectorIndex(self, actionPo, board):
        paramPo = board["param"]
        statePo = board["state"]
        mapData = map.MapData()
        unitData = unit.UnitData()
        map.fromPortable(paramPo["map"], mapData)
        unit.fromPortable(statePo["units"], unitData, mapData)
        # Get hex of actor
        # Get the actor's id
        if actionPo["type"]=="pass":
            return self.getActionSize()-1
        elif actionPo["type"]=="move":
            actorId = actionPo["mover"]
            targetHexId = actionPo["destination"]
            targetHexObj = mapData.hexIndex[targetHexId]
        else: # "fire"
            actorId = actionPo["source"]
            targetUnitId = actionPo["target"]
            targetHexObj = unitData.unitIndex[targetUnitId].hex
        actorObj = unitData.unitIndex[actorId]
        actorHexObj = actorObj.hex
        col_from = actorHexObj.x_offset
        row_from = actorHexObj.y_offset
        col_to = targetHexObj.x_offset
        row_to = targetHexObj.y_offset
        dim = mapData.getDimensions()
        width = dim["width"]
        height = dim["height"]
        n_tgt_hexes = self.getNumTargetHexes()
        hex_code = self.targetCoordsToCode((col_from,row_from),(col_to,row_to))
        index = row_from*width*n_tgt_hexes + col_from*n_tgt_hexes + hex_code
        return index

    def vectorIndexActionToAtlatl(self, actionOffset, board):
        paramPo = board["param"]
        statePo = board["state"]
        mapData = map.MapData()
        unitData = unit.UnitData()
        map.fromPortable(paramPo["map"], mapData)
        unit.fromPortable(statePo["units"], unitData, mapData)
        dim = mapData.getDimensions()
        width = dim["width"]
        height = dim["height"]
        n_actions = self.getActionSize()
        if actionOffset==n_actions-1:
            return {"type":"pass"}
        n_tgt_hexes = self.getNumTargetHexes()
        row = math.floor(actionOffset/(width*n_tgt_hexes))
        rest = actionOffset-row*width*n_tgt_hexes
        col = math.floor(rest/n_tgt_hexes)
        tgt_code = rest - col*n_tgt_hexes
        origin = f'hex-{col}-{row}'
        if not origin in unitData.occupancy:
            return None
        actorObj = unitData.occupancy[origin][0]
        originObj = mapData.hexIndex[origin]
        tgt_coords = self.codeToTargetCoords((col,row),tgt_code)
       
        targetHex = mapData.hexIndex[f'hex-{tgt_coords[0]}-{tgt_coords[1]}']
        if not targetHex:
            return None
        targetHexId = targetHex.id
        targetOccupants = unitData.occupancy.get(targetHexId,[])
        if targetOccupants: # shoot
            targetId = targetOccupants[0].uniqueId
            return {"type":"fire","source":actorObj.uniqueId,"target":targetId}
        else: # move
            return {"type":"move","mover":actorObj.uniqueId,"destination":targetHexId}
            
    def getNextState(self, board, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action as portable Python
            nextPlayer: player who plays in the next turn (should be -player)
        """
        actionPo = self.vectorIndexActionToAtlatl(action, board)
        nextState = self.game.transition(board["state"], actionPo)
        nextBoard = {"param":board["param"], "state":nextState}
        return nextBoard

    def getValidMoves(self, board):
        """
        Input:
            board: current board

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        result = [0]*self.getActionSize()
        # print('result', result)
        statePo = board["state"]
        # print('statePo', statePo)
        actionsJSON = self.game.legal_actions(statePo)
        # print('actionsJSON', actionsJSON)
        # print('len(actionsJSON)', len(actionsJSON))
        for actionPo in actionsJSON:
            index = self.atlatlActionToVectorIndex(actionPo, board)
            action_check = self.vectorIndexActionToAtlatl(index, board)
            # print('actionPo', actionPo)
            # print('index', index)
            result[index] = 1
        #     print('result[index]', result[index])
        # print('result', result)
        return result


    def getIsTerminal(self, board):
        """
        Input:
            board: current board

        Returns:
            True if game is terminal
        """
        statePo = board["state"]
        isTerminal = statePo["status"]["isTerminal"]
        return isTerminal

    def getScore(self, board):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            Was current score normalized to [-1,1]. Now unnormalized raw score.
        """
        statePo = board["state"]
        score = statePo["status"]["score"]
        # maxScoreCI5 = 400 + 240 # Kill all opponents and occupy city every phase
        # score_11 = float(score)/maxScoreCI5
        # maxScoreAtomic = 100 # Kill opposing unit and lose nothing
        max_score = self.game_spec["max_score"]
        score_11 = float(score)/max_score
        #return score_11
        return score

    def _flipFactions(self, board):
        def _flipFaction(faction):
            if faction=="blue":
                return "red"
            elif faction=="red":
                return "blue"
            return faction
        state = board["state"]
        blue_pov_state = copy.deepcopy(state)
        # Fix status
        status = blue_pov_state["status"]
        for city in status["cityOwner"]:
            status["cityOwner"][city] = _flipFaction(status["cityOwner"][city])
        status["score"] = -status["score"]
        status["onMove"] = "blue"
        # Fix units in state
        for unit in blue_pov_state["units"]:
            unit["faction"] = _flipFaction(unit["faction"])
        # Fix units in param
        param = board["param"]
        blue_pov_param = copy.deepcopy(param)
        for unit in blue_pov_param["units"]:
            unit["faction"] = _flipFaction(unit["faction"])
        return {"param":blue_pov_param, "state":blue_pov_state}

    def getCanonicalForm(self, board):
        """
        Input:
            board: current board

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        # Return board if blue is on move
        # Otherwise invert state to be from blue's perspective.
        # Special case: if phase is odd, it will remain odd, even
        #   though odd phases are impossible for blue
        
        if board["state"]["status"]["onMove"]=="blue":
            return board
        return self._flipFactions(board)

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                    form of the board and the corresponding pi vector. This
                    is used when training the neural network from examples.
                    BOARD IS CONVERTED TO A PYTORCH TENSOR
        """
        # FIX: board can be mirrored left to right around its center
        return [(nnetObservation(board),pi)]

    def getPlayerOnMove(self, board):
        """
        Input:
            board: current board

        Returns:
            player: current player (1 or -1)
        """
        if board["state"]["status"]["onMove"]=="blue":
            return 1
        elif board["state"]["status"]["onMove"]=="red":
            return -1
        else:
            raise "onMove player is unknown faction: "+board["state"]["status"]["onMove"]

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                        Required by MCTS for hashing.
        """
        #return json.dumps(board, sort_keys=True)
        return game.statePlusParamHashKey(board["state"], board["param"])

if __name__=="__main__":
    import json
    import game_dispenser
    import scenario
    import random
    scenarioName = "Scenario1middleIsland.scn"
    # scenario_path = os.path.join("/home", "joseph.coble", "Thesis", "Thesis", "atlatl-coble", "server", "scenarios", scenarioName)
    # scenarioPo = json.load(open(scenario_path))
    # scenarioPo = json.load( open("scenarios/"+scenarioName) )
    # scenarioGen = scenario.clear_square_factory(**{'size':5, 'min_units':2, 'max_units':4, 'num_cities':1})
    # scenarioPo = scenarioGen()
    # print('scenarioPo',scenarioName)
    # game = game.Game(scenarioName)
    # print('game',game)
    # ASgame = AtlatlGame(game)
    ASgame = AtlatlGame(scenarioName)
    scenarioPo = ASgame.game.parameters()
    board = ASgame.getInitBoard()

    action2index = {}
    print("Move Index, JSON Action")
    for i in range(883): # 151 made it 300 to test
        result = ASgame.vectorIndexActionToAtlatl(i, board)
        if result:
            print(f'{i} {result}')
            action2index[json.dumps(result, sort_keys=True)] = i
    print("Verifying Result")
    for actionJSON in action2index:
        index1 = action2index[actionJSON]
        actionPo = json.loads(actionJSON)
        index2 = ASgame.atlatlActionToVectorIndex(actionPo, board)
        status = ""
        if index1!=index2:
            status = "FAIL"
        print(f'{index1} {index2} {status}')
    print("\n\nRandom Play Test")
    _player = 0
    done = False
    score = None
    while not done:
        validVec = ASgame.getValidMoves(board, _player)
        validIndices = []
        for i in range(len(validVec)):
            if validVec[i]:
                validIndices.append(i)
        action = random.choice(validIndices)
        print(f'action {action}')
        board, _nextPlayer = ASgame.getNextState(board, _player, action)
        state = board["state"]
        nn_obs = nnetObservation(board)
        print(f'nnet observation {nn_obs}')
        if state["status"]["onMove"] =="red":
            bluePOVState = ASgame._makeStateBluePOV(state)
            print(f'state {state}')
            print(f'bluePOV {bluePOVState}')
        done, score = ASgame.getGameEnded(board, _player)
    print(f'score {score}')
    
