# ByteTrack assignment on ACT-SHAVE

This contribution is under active implementation. It provides the bounded
linear-assignment core used by the ByteTrack integration: FP32 rectangular
cost matrices up to 16 by 16 are solved by a shortest-augmenting-path kernel
on Intel NPU3720. The CPU prepares costs, launches the graph, and validates the
fixed-capacity result; it does not repair or replace the device assignment.

The final video demo and recorded hardware results will be documented here
after they have been validated.
