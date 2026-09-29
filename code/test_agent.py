
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.models import OllamaModel

from agent import document_qa


# ---------------------------------------
# DeepEval Judge Model
# ---------------------------------------

judge_model = OllamaModel(
    model="llama3.2",
    base_url="http://localhost:11434",
    temperature=0
)


# ---------------------------------------
# Test 1: Answer Relevance
# ---------------------------------------

def test_answer_relevance():

    question = "How many days can employees work remotely?"

    actual_output = document_qa(question)

    print("\n" + "=" * 60)
    print("TEST 1: ANSWER RELEVANCE")
    print("=" * 60)
    print("Question:")
    print(question)
    print("\nAgent Answer:")
    print(actual_output)

    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output
    )

    relevance_metric = GEval(
        name="Answer Relevance",
        criteria="""
    The user's question asks: "How many days can employees work remotely?"

    The correct information from the document is:
    "Employees are allowed to work remotely up to three days per week."

    Give a high score if the actual output clearly states this fact.
    Give a low score only if the answer gives a different number,
    says the information is unavailable, or does not answer the question.

    Do NOT evaluate whether the employee is currently working remotely.
    Do NOT evaluate the current time.
    Only evaluate whether the answer correctly addresses the question.
    """,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT
        ],
        threshold=0.7,
        model=judge_model
    )

    print("\nEvaluating with DeepEval...")
    print("Threshold: 0.70")

    assert_test(
        test_case,
        [relevance_metric]
    )

    # Calculate and display the metric
    result = relevance_metric.measure(test_case)

    print(f"Score: {result:.2f}")

    if result >= 0.0:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")


# ---------------------------------------
# Test 2: Answer Correctness
# ---------------------------------------

def test_answer_correctness():

    question = "What are the standard working hours?"

    expected_output = (
        "The standard working hours are from 9:00 AM to 6:00 PM."
    )

    actual_output = document_qa(question)

    print("\n" + "=" * 60)
    print("TEST 2: ANSWER CORRECTNESS")
    print("=" * 60)
    print("Question:")
    print(question)

    print("\nExpected Answer:")
    print(expected_output)

    print("\nAgent Answer:")
    print(actual_output)

    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        expected_output=expected_output
    )

    correctness_metric = GEval(
        name="Answer Correctness",
         criteria="""
    Compare the actual output with the expected output.

    Expected answer:
    "The standard working hours are from 9:00 AM to 6:00 PM."

    The actual output should state that the standard working hours
    are 9:00 AM to 6:00 PM.

    Give a high score if the actual output contains the correct
    working hours, even if the wording is different.

    Give a low score only if the working hours are incorrect,
    missing, or contradicted.

    Do NOT evaluate the current time.
    Do NOT evaluate whether the user is currently working.
    Do NOT require additional information beyond the expected answer.
    """,
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT
        ],
        threshold=0.5,
        model=judge_model
    )

    print("\nEvaluating with DeepEval...")
    print("Threshold: 0.50")

    assert_test(
        test_case,
        [correctness_metric]
    )

    # Calculate and display the metric
    result = correctness_metric.measure(test_case)

    print(f"Score: {result:.2f}")

    if result>= 0.0:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")