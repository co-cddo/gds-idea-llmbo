from llmbo.models import Manifest


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
