from src.live_predict import live_prediction

if __name__ == "__main__":
    result = live_prediction(interval="5m")

    print("\n📈 Live Prediction:")
    for k, v in result.items():
        print(f"{k}: {v}")