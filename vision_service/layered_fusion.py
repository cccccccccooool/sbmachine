"""冷藏区 VLM 分层融合工具的兼容别名，实现转发至 vlm.vision_service.layered_fusion。"""
import sys

from vlm.vision_service import layered_fusion as _implementation

sys.modules[__name__] = _implementation
