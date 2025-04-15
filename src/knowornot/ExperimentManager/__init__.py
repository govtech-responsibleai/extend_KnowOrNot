from ..SyncLLMClient import SyncLLMClient


class ExperimentManager:
    def __init__(self, default_client: SyncLLMClient):
        self.default_client = default_client
