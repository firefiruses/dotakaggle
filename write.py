import pandas as pd
def write(model):
    writeData = pd.read_csv("DOTA2_TEST_features.csv")
    ids = writeData["match_id"]
    writeData = writeData.drop("match_id", axis = 1)
    resultWriteData = model.predict(writeData)
    resultWriteData = pd.DataFrame({"match_id" : ids, "radiant_win" : resultWriteData})
    resultWriteData.to_csv("result.csv", index= False)