from llmbo.models import Manifest, ModelInput


def test_basic_instantiation():
    """Test that a normal instantiation works correctly."""
    manifest = Manifest(
        totalRecordCount=100,
        processedRecordCount=90,
        successRecordCount=85,
        errorRecordCount=5,
        inputTokenCount=1_000_000,
        outputTokenCount=2_000_000,
    )
    assert manifest.totalRecordCount == 100
    assert manifest.processedRecordCount == 90
    assert manifest.successRecordCount == 85
    assert manifest.errorRecordCount == 5
    assert manifest.inputTokenCount == 1_000_000
    assert manifest.outputTokenCount == 2_000_000


def test_optional_instantiation():
    """Test that a normal instantiation works correctly.

    Manifests that fail entirely do not have TokenCounts. The model should be created regardless"
    """
    manifest = Manifest(
        totalRecordCount=100,
        processedRecordCount=90,
        successRecordCount=85,
        errorRecordCount=5,
    )
    assert manifest.totalRecordCount == 100
    assert manifest.processedRecordCount == 90
    assert manifest.successRecordCount == 85
    assert manifest.errorRecordCount == 5
    assert manifest.inputTokenCount is None
    assert manifest.outputTokenCount is None


def test_manifest_allows_unknown_fields():
    """Test that Manifest accepts unknown fields via Pydantic extra='allow'.

    AWS Bedrock may return new fields (e.g. inputAudioSecond) that are not
    defined on the Manifest model. These should be captured in model_extra
    rather than raising a validation error.
    """
    manifest = Manifest(
        totalRecordCount=200,
        processedRecordCount=180,
        successRecordCount=170,
        errorRecordCount=10,
        inputTokenCount=500_000,
        outputTokenCount=1_000_000,
        inputAudioSecond=42,
        someNewField="unexpected",
    )

    assert manifest.totalRecordCount == 200
    assert manifest.processedRecordCount == 180
    assert manifest.successRecordCount == 170
    assert manifest.errorRecordCount == 10
    assert manifest.inputTokenCount == 500_000
    assert manifest.outputTokenCount == 1_000_000
    assert manifest.model_extra == {
        "inputAudioSecond": 42,
        "someNewField": "unexpected",
    }


def test_modelinput_to_dict_with_tool_choice_model():
    """Test that a normal instantiation works correctly.

    Manifests that fail entirely do not have TokenCounts. The model should be created regardless"
    """
    mi = ModelInput(messages=[], tool_choice="any")
    mi.to_dict()
