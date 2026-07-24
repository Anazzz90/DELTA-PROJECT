from core.pipeline import Pipeline
print("Initializing Pipeline...")
p = Pipeline()
print("Pipeline initialized!")
print(f"Agents: {[a.name for a in p.agents]}")
