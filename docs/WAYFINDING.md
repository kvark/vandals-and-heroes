# Mission wayfinding pass

Vandals and Heroes already combines curved terrain, a physics-driven mechos, and
a short radio-led chase followed by two deliveries. The depot and ridge cache
have distinguishable beacon colors and a planner finds climbable routes on the
shipped terrain. However, the player-facing title and web HUD pointed at the
final beacon even when the planned route deliberately went around a cliff.
The autopilot had access to information the human driver did not.

- [K-D Lab's Vangers](https://kdlab.com/vangers/en) is the reference for the
  vehicle-led alien-world identity. Keep the trip and terrain central; use
  field-sign graphics rather than fantasy quest medallions.
- [Focus Entertainment's SnowRunner](https://www.focus-entmt.com/en/games/snowrunner)
  is the reference for driving difficulty that comes from terrain and route
  selection. Give the driver a readable next direction without taking control
  or smoothing out the terrain.

The new road, depot and ridge SVG emblems are original vector artwork authored
for this pass. Their road, crate and peak shapes communicate the active phase
even without the amber, cyan and violet color cues. They are inline in the
existing page, with no secondary network or asset-pack dependency. The same
mission state still provides the accessible text.

During a planned delivery, the title/HUD keeps the final beacon's straight-line
distance and uses the bearing to the next route point. Near the claim radius, or
if no applicable route exists, it falls back to the previous direct guidance.
It is a hint along the assigned route, not live replanning after a detour. The
change affects neither mission triggers and rewards nor steering and physics.

The two in-world landmarks are new procedural meshes: a low, strapped depot
crate and a tall ridge cairn with a diamond crown. Their silhouettes match the
mission artwork. Both render only while their objective is active, are aligned
to the terrain's local up direction, and are visual-only so they cannot snag
the car or alter collection behavior. Existing beacon lights remain.
