[![Build Status](https://app.travis-ci.com/pauleaster/vertex_finder.svg?branch=main)](https://app.travis-ci.com/pauleaster/vertex_finder)

# Vertex Finder
## Description
This code was written in rust as an aid to learning the language. The code places `n` vertices on a unit sphere and calculates the approximate Coulomb force to determine the direction of movement for each vertex. The vertex position is evolved in time until the vertices reach a stable position. This does not implement a full Hamiltonian or Lagrangian evolution as the direction for movement is evaluated and initial velocity is calculated bu the velocity is discarded after the vertex is repositioned. After the vertices reach a steady state then the outer faces are calculated from the edges between the vertices. Eventually the image as seen from a specific camera direction will be calculated and this image will be rotated in three dimensional space and shown on my website.
## Methodology
+ External crates were minimised to aid in the learning process with Rust. This enabled me to learn some of the intricacies involved with Rust including borrowing and moving.  
+ This code has not had a full testing implemented, most testing has been performed with printing and commenting.
+ Note that because of the performance of the Rust compiler, the amount of debugging needed for this code was surprising low! In many cases, if it compiles it will run correctly! Any logical bug were quite easy to isolate and fix using the debugger in VScode.
+ Another surprising thing that I found about writing Rust code was that it was still relatively easy to prototype the code and build incrementally. I did not expect this to be the case!