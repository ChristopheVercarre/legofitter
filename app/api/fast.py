from fastapi import FastAPI


app = FastAPI(
    title="LegoFitter API",
    version="0.1.0"
)


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "LegoFitter API"
    }


@app.get("/ping")
def ping():
    return {
        "message": "pong"
    }
