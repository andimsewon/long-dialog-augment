import argparse
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from config import MAX_NEW_TOKENS, MODEL_ID, TARGET_TURNS, TEMPERATURE, WINDOW_SIZE
from dialog_parser import flatten_sessions
from llm_generator import build_prompt, generate_next_turn, load_template
from utils import count_turns, ensure_dir, load_json, save_json

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 200  # absolute ceiling to prevent infinite loops

# Matches "speaker1: ..." or "Speaker2: ..." case-insensitively.
_TURN_RE = re.compile(r"^(speaker\d+)\s*:\s*(.+)$", re.IGNORECASE)


def parse_generated_text(
    text: str,
    known_speakers: Optional[set] = None,
) -> List[Dict[str, str]]:
    """
    Extract speaker turns from raw LLM output.

    Filters to lines that match `speaker<N>: <utterance>`. If known_speakers
    is provided, lines whose speaker label is not in the set are dropped —
    this prevents LLM-generated prose or header lines from leaking in.
    """
    turns = []
    for line in text.strip().splitlines():
        m = _TURN_RE.match(line.strip())
        if not m:
            continue
        speaker = m.group(1).lower()
        utterance = m.group(2).strip()
        if known_speakers and speaker not in known_speakers:
            continue
        turns.append({"speaker": speaker, "utterance": utterance})
    return turns


def extract_persona_and_profile(json_data: dict):
    cl = json_data["participantsInfo"]["speaker1"]
    cp = json_data["participantsInfo"]["speaker2"]
    cl_feats = json_data["personaInfo"]["clInfo"]["personaFeatures"]
    cp_feats = json_data["personaInfo"]["cpInfo"]["personaFeatures"]

    persona1 = " ".join(cl_feats)
    persona2 = " ".join(cp_feats)
    profile1 = (
        f"{cl['gender']} / {cl['age']} / {cl['occupation']} / "
        f"{cl['educationLevel']} / 출신: {cl['bPlace']}"
    )
    profile2 = (
        f"{cp['gender']} / {cp['age']} / {cp['occupation']} / "
        f"{cp['educationLevel']} / 출신: {cp['bPlace']}"
    )
    return persona1, persona2, profile1, profile2


def extract_persona_summary(json_data: dict) -> str:
    """Build a compact persona summary from the last session's accumulated notes."""
    sessions = json_data.get("sessionInfo", [])
    if not sessions:
        return ""
    summary = sessions[-1].get("sessionPersonaSummary", {})
    lines = [
        f"{speaker}: " + " / ".join(notes)
        for speaker, notes in summary.items()
        if notes
    ]
    return "\n".join(lines)


def augment_file(
    input_path: str,
    output_path: str,
    topic: str,
    template: str,
) -> None:
    raw_data = load_json(input_path)
    persona1, persona2, profile1, profile2 = extract_persona_and_profile(raw_data)
    persona_summary = extract_persona_summary(raw_data)
    dialog = flatten_sessions(raw_data)
    known_speakers = {"speaker1", "speaker2"}

    logger.info("Augmenting %s  (seed turns: %d)", input_path, count_turns(dialog))

    pbar = tqdm(
        total=TARGET_TURNS,
        initial=count_turns(dialog),
        unit="turn",
        desc=Path(input_path).stem,
        leave=True,
    )

    for iteration in range(MAX_ITERATIONS):
        if count_turns(dialog) >= TARGET_TURNS:
            break

        prev_len = len(dialog)
        prompt = build_prompt(
            dialog,
            topic=topic,
            persona1=persona1,
            persona2=persona2,
            profile1=profile1,
            profile2=profile2,
            template=template,
            persona_summary=persona_summary,
        )
        generated = generate_next_turn(prompt)
        new_turns = parse_generated_text(generated, known_speakers=known_speakers)
        dialog.extend(new_turns)

        added = count_turns(dialog) - (prev_len // 2)
        if added > 0:
            pbar.update(added)

        if len(dialog) == prev_len:
            logger.warning(
                "Iteration %d produced no new turns; stopping early.", iteration + 1
            )
            break
    else:
        logger.warning("Reached MAX_ITERATIONS (%d) without hitting target.", MAX_ITERATIONS)

    pbar.close()

    output = {
        "metadata": {
            "source_file": os.path.basename(input_path),
            "topic": topic,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_utterances": len(dialog),
            "total_turns": count_turns(dialog),
            "config": {
                "target_turns": TARGET_TURNS,
                "window_size": WINDOW_SIZE,
                "temperature": TEMPERATURE,
                "max_new_tokens": MAX_NEW_TOKENS,
                "model": MODEL_ID,
            },
        },
        "dialog": dialog,
    }

    ensure_dir(os.path.dirname(output_path))
    save_json(output, output_path)
    logger.info("Saved %d turns → %s", count_turns(dialog), output_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Augment Korean multi-session dialogues to 60+ turns using LLaMA."
    )
    parser.add_argument("--topic", required=True, help="Conversation topic (e.g. '호캉스')")
    parser.add_argument(
        "--template",
        default="templates/prompt_template.txt",
        help="Path to the prompt template file",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", help="Single input JSON file")
    mode.add_argument("--input-dir", help="Directory of input JSON files (batch mode)")

    parser.add_argument(
        "--output",
        help="Output path for single-file mode (required with --input)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/augmented",
        help="Output directory for batch mode (default: outputs/augmented)",
    )

    args = parser.parse_args()

    if args.input and not args.output:
        parser.error("--output is required when using --input")

    template = load_template(args.template)

    if args.input:
        augment_file(args.input, args.output, args.topic, template)
    else:
        input_files = sorted(Path(args.input_dir).glob("*.json"))
        if not input_files:
            logger.error("No JSON files found in %s", args.input_dir)
            return
        logger.info("Batch mode: %d files in %s", len(input_files), args.input_dir)
        for path in input_files:
            out = os.path.join(args.output_dir, path.stem + "-augmented.json")
            try:
                augment_file(str(path), out, args.topic, template)
            except Exception as e:
                logger.error("Failed on %s: %s", path.name, e)


if __name__ == "__main__":
    main()
