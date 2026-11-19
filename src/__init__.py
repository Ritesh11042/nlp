"""
Adaptive Preprocessing Selection for Domain-Specific NLP Pipelines
A Meta-Learning Approach

Modules:
- DomainPolicyEngine: 4 specialized preprocessing policies (P0-P3)
- LocalMetaProfiler: 9-D label-free meta-vector profiler
- MicroClusterPartitioner: MiniLM embedding + Spherical K-Means clustering
- LightGBMSurrogateRouter: Calibrated surrogate policy router
- AdaptivePreprocessingGateway: End-to-end gateway orchestrator
- ExtendedEmojiEngine: 80+ emoji and emoticon decoder
- WordIntelligenceEngine: Differentiates Right vs Wrong vs Domain Entity words, spelling correction & missing word restoration
"""

from .policy_engine import DomainPolicyEngine, POLICY_NAMES, POLICY_DESCRIPTIONS
from .profiler import LocalMetaProfiler
from .partitioner import MicroClusterPartitioner
from .router import LightGBMSurrogateRouter
from .gateway import AdaptivePreprocessingGateway
from .word_intelligence import ExtendedEmojiEngine, WordIntelligenceEngine

__all__ = [
    "DomainPolicyEngine",
    "POLICY_NAMES",
    "POLICY_DESCRIPTIONS",
    "LocalMetaProfiler",
    "MicroClusterPartitioner",
    "LightGBMSurrogateRouter",
    "AdaptivePreprocessingGateway",
    "ExtendedEmojiEngine",
    "WordIntelligenceEngine",
]
