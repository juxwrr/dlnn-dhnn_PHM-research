# Changes for the Windows transfer

The export is a separate checkout. It does not alter the running Mac programs or their frozen snapshots.

Portability changes: replace mandatory POSIX `resource` import with a platform helper; normalize Linux KiB versus Mac bytes for memory reporting; use the writable system temporary directory for Matplotlib; locate original reference material relative to the repository; provide UTF-8 reads and Windows timezone data; add PowerShell setup/start scripts and a portable study coordinator.

Continuation changes: preserve saved optimizer state, RNG position, acceleration scales, validation data, best checkpoint and integer step. Permit only roundoff-level differences when reconstructing the coordinate scale constants on another backend, then retain the stored transform exactly. The full 300,000-update learning-rate schedule is not restarted. A different hardware backend may produce floating-point differences.

Unchanged scientific components: network layout, Softplus activation, custom initialization, logical batch 512, chunk sizes, optimizer objective and update count, physical reference/action terms, pseudoinverse method, precision, seed/split protocol, source control interpretation, no extra losses or clipping, failed-case reporting and physical full-model gate.

New verification checks exercise cross-platform imports, the relative archived reference, argument preservation and actual optimizer resumption versus uninterrupted training. Windows and Linux CI also run the prior recipe/equation checks and physics audit. AMD GPU compatibility is explicitly a separate hardware check.
