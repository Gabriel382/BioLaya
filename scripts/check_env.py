from __future__ import annotations
import json
from biolaya.cloud.env import environment_report
print(json.dumps(environment_report(), indent=2))
