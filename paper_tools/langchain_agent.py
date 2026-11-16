"""Run the paper-reproduction tools through a LangChain agent.

This file is intentionally separate from the tool implementations.  The tools
remain useful without an LLM, while this module supplies the LangChain agent
that chooses the order in which to call them.

Requires the optional packages::

    python -m pip install -U langchain langchain-openai

Set OPENAI_API_KEY before running the example, or replace ChatOpenAI below with
another LangChain chat-model integration.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .docker_build import make_tools as make_build_tools
from .read_tools import read_file, read_paper
from .reproduce_tools import make_tools as make_reproduce_tools
from .spec_validation_tools import make_tools as make_spec_tools
from .state import ReproState


SYSTEM_PROMPT = """
You are a paper-reproduction assistant. Use the supplied tools and preserve
the shared reproduction state.
The runner preloads the exact specification selected on the command line.
Use that active specification; do not invent a different path.

The command-line runner has already loaded the exact paper specification.
Do not call load_paper_spec again and do not invent another specification path.
Your job is to execute the reproduction, not merely describe the steps. Emit
actual tool calls and wait for their returned results. Do not say that you are
"executing" a tool unless a tool call is present in the response.

For a reproduction request, follow this order:
1. Use the already-loaded active specification.
2. If the active spec uses the Slurm backend, call reproduce_metric directly;
   it submits the configured Slurm job and waits for its metrics. Do not call
   docker_build for a Slurm spec.
3. If the active spec uses the auto backend, call docker_build first. The
   reproduce tool will run a real CUDA operation in the local image; it uses
   Docker locally only if that probe succeeds, otherwise it submits the Slurm
   fallback.
4. Otherwise, call docker_build when the requested image is not already available.
5. Call reproduce_metric for one metric at a time; do not issue multiple
   reproduce_metric calls in parallel because they share state and Docker
   resources.
6. Call validate_metric for every metric that reproduced successfully, one at
   a time.
7. Report the actual status returned by each tool. A reproduction is complete
   only when the metrics have status OK and the corresponding validations have
   status PASS (or NO_RECORD when the spec intentionally has no reference).
8. Do not treat a smoke-spec result as a scientific paper reproduction. Use
   the full paper spec when the user requests the full experiment.
9. When the active spec is named UAP-paper-faithful-CIFAR10, its command is
   authoritative. Do not replace it with the reduced demo, reuse the old
   21-model checkpoints, lower the number of models/points/views/epochs, or
   report metrics before the command has completed.
10. If a long-running reproduce_metric call returns RUN_FAILED or ERROR, stop
   requesting further metrics for that same specification. Report the failure
   and its evidence log instead of automatically launching the same expensive
   command again.

Use the smoke spec for a fast plumbing test. Do not substitute the smoke
results for the scientific UAP results. The full UAP spec may run for hours
and requires the trained checkpoints and a GPU.
Return the status, extracted values, validation results, and evidence-log
paths clearly.
"""


def build_agent(model, workspace: str):
    """Create an agent whose tools all share one ReproState instance."""
    try:
        from langchain.agents import create_agent
    except ImportError as exc:  # pragma: no cover - exercised only without optional deps
        raise RuntimeError(
            "Install the optional agent dependencies first: "
            "python -m pip install -U langchain langchain-openai"
        ) from exc

    state = ReproState(workspace)
    tools = []
    # The CLI preloads the exact spec path before the agent starts. Expose
    # validation, but not a second model-controlled spec loader, so a model
    # cannot replace the verified Windows path with an invented one.
    tools.extend(make_spec_tools(state)[1:])
    tools.extend(make_build_tools(state))
    tools.extend(make_reproduce_tools(state))
    tools.extend([read_file, read_paper])

    agent = create_agent(model=model, tools=tools, system_prompt=SYSTEM_PROMPT)
    return agent, state, tools


def build_bound_tools(workspace: str):
    """Create the same shared LangChain tools without constructing an LLM."""
    state = ReproState(workspace)
    tools = []
    tools.extend(make_spec_tools(state)[1:])
    tools.extend(make_build_tools(state))
    tools.extend(make_reproduce_tools(state))
    tools.extend([read_file, read_paper])
    return state, tools


def _tool_json(tool, arguments: dict) -> dict:
    result = tool.invoke(arguments)
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"status": "ERROR", "message": result}
    return result


def run_deterministic_workflow(state, tools) -> dict:
    """Execute the bound LangChain tools in a fixed reproduction order.

    This is a safety net for models that describe a tool call in plain text
    instead of returning an actual tool-call message. The work is still
    performed by the same LangChain tool objects and shared ReproState.
    """
    by_name = {tool.name: tool for tool in tools}
    output = {"build": None, "metrics": {}, "evidence_directory": state.workspace}
    if state.spec.backend in {"docker", "auto"}:
        image = state.spec.image_tag()
        previous_build = state.data.get("images", {}).get(image, {})
        if previous_build.get("ok"):
            output["build"] = {"status": "OK", "image": image, "reused_previous_build": True}
        else:
            output["build"] = _tool_json(
                by_name["docker_build"],
                {"dockerfile": state.spec.dockerfile},
            )
        if output["build"].get("status") != "OK" and state.spec.backend == "docker":
            output["status"] = "FAILED"
            return output

    failed = False
    for metric in state.spec.metrics:
        reproduced = _tool_json(by_name["reproduce_metric"], {"metric": metric})
        record = {"reproduce": reproduced, "validate": None}
        if reproduced.get("status") == "OK":
            record["validate"] = _tool_json(by_name["validate_metric"], {"metric": metric})
            failed |= record["validate"].get("status") not in {"PASS", "NO_RECORD"}
        else:
            failed = True
            output["metrics"][metric] = record
            break
        output["metrics"][metric] = record
    output["status"] = "FAILED" if failed else "OK"
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Run UAP reproduction tools via LangChain")
    parser.add_argument(
        "--spec",
        default="papers/uap/tool_smoke_spec.yaml",
        help="paper spec to load; default is the fast tool smoke spec",
    )
    parser.add_argument(
        "--workspace",
        default="inspection-results/langchain-uap-run",
        help="directory where state and logs are persisted",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LANGCHAIN_MODEL", "gpt-4o-mini"),
        help="chat model name for the selected provider",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="optional custom instruction; otherwise runs all three smoke metrics",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="run the bound LangChain tools in a fixed order without an LLM/API key",
    )
    args = parser.parse_args()

    spec_path = str(Path(args.spec).resolve())

    if args.deterministic:
        state, tools = build_bound_tools(args.workspace)
        try:
            state.set_spec(spec_path)
        except Exception as exc:
            raise SystemExit(f"Could not load spec '{spec_path}': {type(exc).__name__}: {exc}") from exc
        print(json.dumps(run_deterministic_workflow(state, tools), indent=2, default=str))
        print(f"Evidence directory: {state.workspace}")
        return 0

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise SystemExit(
            "Install the optional agent dependencies first:\n"
            "  python -m pip install -U langchain langchain-openai"
        ) from exc

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Set it before running the agent.")

    model = ChatOpenAI(model=args.model, temperature=0)
    agent, state, tools = build_agent(model, args.workspace)
    try:
        # Load the exact CLI-selected spec before the model starts. This keeps
        # Windows paths with backslashes from being misread by the LLM and
        # guarantees that every bound tool has an active specification.
        state.set_spec(spec_path)
    except Exception as exc:
        raise SystemExit(f"Could not load spec '{spec_path}': {type(exc).__name__}: {exc}") from exc
    if args.prompt:
        prompt = args.prompt
    elif state.spec.name == "UAP-paper-faithful-CIFAR10":
        prompt = (
            f"The active paper-faithful specification is already loaded from this exact path: {spec_path}. "
            "Build the image if needed, execute the complete 81-model CIFAR-10 UAP reproduction, "
            "including adversarial soft-label extraction, UAP generation, fingerprint views, "
            "encoder training, every metric, and validation. Do not substitute the reduced demo."
        )
    else:
        prompt = (
            f"The active specification is already loaded from this exact path: {spec_path}. "
            "Test the UAP reproduction tools using that active specification. Build the Docker image "
            "if needed, then reproduce and validate every metric in the active specification."
        )

    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    # Some providers return a natural-language promise such as “Executing the
    # tool now” without an actual tool call. If no run occurred, execute the
    # same bound tools deterministically instead of claiming completion.
    required_keys = {state.key(metric) for metric in state.spec.metrics}
    reproduced_keys = set(state.data.get("reproduced", {}))
    validation_keys = set(state.data.get("validations", {}))
    complete = required_keys.issubset(reproduced_keys) and required_keys.issubset(validation_keys)
    if not complete:
        print("The LangChain response did not complete every metric tool call; finishing with the bound tools directly.")
        print(json.dumps(run_deterministic_workflow(state, tools), indent=2, default=str))
    else:
        messages = result.get("messages", [])
        if messages:
            print(messages[-1].content)
        else:
            print(result)
    print(f"Evidence directory: {state.workspace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
