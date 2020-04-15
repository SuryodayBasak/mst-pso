import data_api as da
from sklearn.model_selection import train_test_split

fdct = {1: da.AnuranCallsFamily(),\
        2: da.AnuranCallsGenus(),\
        3: da.AnuranCallsSpecies(),\
        4: da.AuditRisk(),\
        5: da.Avila(),\
        6: da.BankNoteAuth(),\
        7: da.BloodTransfusion(),\
        8: da.BreastCancer(),\
        9: da.BreastTissue(),\
        10: da.BurstHeaderPacket(),\
        11: da.CSection(),\
        12: da.CardioOtgMorph(),\
        13: da.CardioOtgFetal(),\
        14: da.DiabeticRetino(),\
        15: da.Ecoli(),\
        16: da.Electrical(),\
        17: da.EEGEye(),\
        18: da.Glass(),\
        19: da.Haberman(),\
        20: da.HTRU2(),\
        21: da.ILPD(),\
        22: da.Immunotherapy()}

for i in range (1, 23):
    print("i = ", i)
    dataset = fdct[i]
    X, y = dataset.Data()
    print("X :")
    print(X)
    print("y :")
    print(y)
    print("Splitting training and test sets:")
    X_train, X_test, y_train, y_test = train_test_split(X, y,\
                                       test_size=0.33, random_state=42)
