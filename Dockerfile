FROM python:3.10-slim

RUN apt-get update && apt-get install -y git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*




WORKDIR /app

COPY . .
RUN python3.10 -m pip install --ignore-installed -e .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

RUN python3.10 -m pip install httpx requests pydantic-settings pandas numpy scikit-learn rdkit torch sqlalchemy
