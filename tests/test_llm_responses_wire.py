from tool_relevance_lab.dataset_generation.llm import _responses_text, _responses_text_format


def test_responses_text_prefers_output_text() -> None:
    assert _responses_text({"output_text": '{"ok": true}'}) == '{"ok": true}'


def test_responses_text_falls_back_to_output_chunks() -> None:
    raw = {
        "output": [
            {
                "content": [
                    {"type": "output_text", "text": '{"a":'},
                    {"type": "output_text", "text": "1}"},
                ]
            }
        ]
    }

    assert _responses_text(raw) == '{"a":1}'


def test_responses_text_format_maps_json_object() -> None:
    assert _responses_text_format({"type": "json_object"}) == {"type": "json_object"}
