# Native I/O

In this example, we will show how to write the mesh and results to a file, and later load them, in the native L3STER format.
This functionality is key for long-running simulations.
If your program crashes, you can resume from an intermediate checkpoint - you don't have to rerun the simulation from the beginning.
Some key features of native I/O in L3STER include:

- Separate files for mesh and results - no need to duplicate the mesh if it is constant.
- Unified output - all MPI ranks write to the same file. This is helpful in environments where creating large numbers of files can be a problem (e.g. HPC clusters).
- Flexibility - the number of MPI ranks reading and writing the files can differ. If it is the same, you can save time by reusing the stored partitioning.