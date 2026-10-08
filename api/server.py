from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Q-Core Optimization API", version="1.0")

@app.get("/")
def read_root():
    return {"message": "Q-Core Quantum Pruning API Aktif"}

@app.post("/compress")
def compress_model():
    # İleride müşterilerin modellerini burada karşılayıp QuantumPruner'a yollayacağız
    return {"status": "success", "message": "Model optimizasyon kuyruğuna alındı."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)