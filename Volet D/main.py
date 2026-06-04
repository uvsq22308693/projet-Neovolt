from ingestion import load_data
from preprocessing import preprocess
from training import train

def main():

    consommation, compteurs, clients, fraudes = load_data()

    df = preprocess(consommation, compteurs, clients, fraudes)

    model, results = train(df)

    print("Pipeline terminé ✔")
    print(results)

if __name__ == "__main__":
    main()