# core/orchestrator.py

class Orchestrator:
    def __init__(self, agents: dict):
        self.agents = agents

    def handle_task(self, task_type: str, input_data: str):
        if task_type in self.agents:
            return self.agents[task_type].run(input_data)
        else:
            return f"No agent available for task '{task_type}'"
