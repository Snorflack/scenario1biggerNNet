IMPORT_NEURAL = True

import ai.passive
import ai.random_actor
import ai.shootback
import ai.potential_field
import ai.dijkstra_demo
import ai.mcts
import ai.gym_ai_surrogate
import ai.pass_agg
import ai.pass_agg_fog
import ai.burt_reynolds_lab2
import ai.burtplus
if IMPORT_NEURAL:
    import ai.neural
    import ai.azero

ai_registry = {
              "passive" : (ai.passive.AI, {}),
              "random" : (ai.random_actor.AI, {}),
              "shootback" : (ai.shootback.AI, {}),
              "field" : (ai.potential_field.AI, {}),
              "pass-agg" : (ai.pass_agg.AI, {}),
              "pass-agg-fog" : (ai.pass_agg_fog.AI, {}),
              "dijkstra" : (ai.dijkstra_demo.AI, {}),

              "mcts1k" : (ai.mcts.AI, {"max_rollouts":1000, "debug":False}),
              "mcts10k" : (ai.mcts.AI, {"max_rollouts":10000, "debug":False}),
              "mcts20k" : (ai.mcts.AI, {"max_rollouts":20000, "debug":False}),
              "mcts30k" : (ai.mcts.AI, {"max_rollouts":30000, "debug":False}),
              "mcts50k" : (ai.mcts.AI, {"max_rollouts":50000, "debug":False}),
              "mctsd" : (ai.mcts.AI, {"max_rollouts":10000, "debug":True}),

              "gym" : (ai.gym_ai_surrogate.AI, {}),
              "gymx2" : (ai.gym_ai_surrogate.AIx2, {}),
              "gym12" : (ai.gym_ai_surrogate.AITwelve, {}),
              "gym13" : (ai.gym_ai_surrogate.AI13, {}),
              "gym14" : (ai.gym_ai_surrogate.AI14, {}),
              "gym16" : (ai.gym_ai_surrogate.AI16, {}),
              "burt-reynolds-lab2" : (ai.burt_reynolds_lab2.AI, {}),
              "burtplus" : (ai.burtplus.AI, {})
             }
             
if IMPORT_NEURAL:
    ai_registry["neural"] = (ai.neural.AI, {"doubledCoordinates":False})
    ai_registry["cnn"] = (ai.neural.AI, {"doubledCoordinates":True})
    ai_registry["hex12"] = (ai.neural.AITwelve, {"doubledCoordinates":False})
    ai_registry["hex13"] = (ai.neural.AI13, {"doubledCoordinates":False})
    ai_registry["hex14"] = (ai.neural.AI14, {"doubledCoordinates":False})
    ai_registry["hex14dqn"] = (ai.neural.AI14, {"dqn":True, "doubledCoordinates":False})
    ai_registry["mando-fun-lab3"] = (ai.neural.AI14, {"neuralNet":"ai/mandofun_c0.zip", "dqn":True, "doubledCoordinates":False})
    ai_registry["alphazero"] = (ai.azero.AIaz, {"neuralNet":"temp"})