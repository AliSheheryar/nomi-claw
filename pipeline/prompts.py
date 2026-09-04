"""Prompts kept separate so they can be tuned without touching pipeline code."""

ENUMERATION_PROMPT = """You are analyzing a segment of a wedding / mehendi video.
The segment starts at absolute time {t0:.2f}s and ends at {t1:.2f}s in the original video.

List every distinct meaningful moment you see. For EACH moment return ONE JSON object with fields:
  "start":      absolute seconds (float, within [{t0:.2f}, {t1:.2f}])
  "end":        absolute seconds (float, within [{t0:.2f}, {t1:.2f}])
  "label":      3-6 words describing the action
  "emotion":    one of tender | romantic | laughter | argument | neutral | busy
  "importance": integer 1..5  (5 = must keep, 1 = filler)

Focus on:
- family members sitting beside the groom on the sofa
- family members applying mehendi on the groom (one by one — treat each as its own moment)
- playful arguments and reactions
- laughter and smiles
- quiet tender exchanges and close-ups

Return ONLY a JSON array of these objects. No prose, no code fences.
If nothing meaningful happens, return [].
"""
