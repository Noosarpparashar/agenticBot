# agents/registry.py
class AgentRegistry:
    def __init__(self):
        self._agents = {}

    def register(self, name: str, agent):
        """Register an agent under a unique name."""
        if name in self._agents:
            raise ValueError(f"Agent '{name}' already registered")
        self._agents[name] = agent

    def get(self, name: str):
        """Retrieve a registered agent by name."""
        if name not in self._agents:
            raise ValueError(f"Agent '{name}' not found")
        return self._agents[name]

    def list_agents(self):
        """Return a list of registered agent names."""
        return list(self._agents.keys())
