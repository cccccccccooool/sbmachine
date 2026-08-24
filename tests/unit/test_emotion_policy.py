from sbmachine.emotion_policy import EmotionPolicy, blend_intensity, normalize_commentary_emotion, weighted_intensity


def test_blend_intensity_uses_seventy_percent_hard_fact():
    assert blend_intensity(0.8, 0.2) == 0.62


def test_llmb_cannot_create_high_emotion_without_hard_facts():
    policy = EmotionPolicy(min_history=12)
    decision = policy.decide(hard_intensity=0.0, llmb_intensity=1.0, scream_eligible=False)
    assert decision.label == "平述"
    assert decision.score == 0.3


def test_scream_requires_hard_fact_eligibility():
    policy = EmotionPolicy(min_history=12)
    blocked = policy.decide(hard_intensity=1.0, llmb_intensity=1.0, scream_eligible=False)
    allowed = policy.decide(hard_intensity=1.0, llmb_intensity=1.0, scream_eligible=True)
    assert blocked.label == "激动"
    assert allowed.label == "惊叹"


def test_short_match_does_not_require_two_hundred_scenes():
    policy = EmotionPolicy(history_size=200, min_history=12)
    for _ in range(3):
        policy.decide(hard_intensity=0.0, llmb_intensity=0.0, scream_eligible=False)
    assert policy.sample_count == 3
    assert policy.history_ready is False


def test_normalize_commentary_clamps_inner_labels_to_final():
    text = "[紧张]先架住。[惋惜]没接住。[惊叹]又打回来了！"
    assert normalize_commentary_emotion(text, "激动") == "[激动]先架住。[平述]没接住。[激动]又打回来了！"


def test_normalize_commentary_clamp_examples():
    assert normalize_commentary_emotion("[平述]包下了[惊叹]三杀！", "激动") == "[激动]包下了[激动]三杀！"
    assert normalize_commentary_emotion("没有标签的文本", "平述") == "[平述]没有标签的文本"
    assert normalize_commentary_emotion("[激动]a", "惊叹") == "[惊叹]a"
    assert normalize_commentary_emotion("[激动][激动]a", "激动") == "[激动]a"


def test_weighted_intensity_uses_scene_duration():
    assert weighted_intensity([(0.2, 2.0), (0.8, 6.0)]) == 0.65
