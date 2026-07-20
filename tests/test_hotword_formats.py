import pytest

from src.core.hotwords.loader import (
    InvalidHotwordFormatError,
    format_hotwords_for_model,
    parse_hotwords,
)


def test_parse_hotwords_accepts_only_strict_json_and_formats_for_current_model():
    hotwords = parse_hotwords('[{"weight":100,"hotword":"篮子"},{"weight":80,"hotword":"直播"}]')

    assert hotwords == [
        {"weight": 100, "hotword": "篮子"},
        {"weight": 80, "hotword": "直播"},
    ]
    assert format_hotwords_for_model(hotwords) == "篮子 直播"
    assert format_hotwords_for_model(hotwords, weighted=True) == [[100, "篮子"], [80, "直播"]]


def test_parse_hotwords_accepts_empty_array():
    assert parse_hotwords("[]") == []


@pytest.mark.parametrize(
    "value",
    [
        "达摩院:30",
        "not-json",
        "",
        "   ",
        "{}",
        '[{"weight":80}]',
        '[{"hotword":"词"}]',
        '[{"weight":80,"hotword":"词","extra":true}]',
        '[{"weight":true,"hotword":"词"}]',
        '[{"weight":0,"hotword":"词"}]',
        '[{"weight":101,"hotword":"词"}]',
        '[{"weight":80.0,"hotword":"词"}]',
        '[{"weight":80,"hotword":""}]',
        '[{"weight":80,"hotword":"   "}]',
        [{"weight": 80, "hotword": "词"}],
    ],
)
def test_parse_hotwords_rejects_non_strict_input(value):
    with pytest.raises(InvalidHotwordFormatError, match="JSON 数组"):
        parse_hotwords(value)
