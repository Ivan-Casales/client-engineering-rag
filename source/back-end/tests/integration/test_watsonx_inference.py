import pytest
from tests.test_utility.utils_watsonx import generate_answer_with_context

@ pytest.mark.integration
def test_generate_answer_with_context():
    context = "Watsonx.ai is IBM’s next-generation AI and data platform."
    question = "What is Watsonx.ai used for?"
    answer = generate_answer_with_context(context, question)
    assert isinstance(answer, str), "Answer should be a string"
    assert len(answer) > 0, "Answer should not be empty"