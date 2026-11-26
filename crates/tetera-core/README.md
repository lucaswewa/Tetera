# tetera-core

Tetera's foundation layer, L0: the types every other crate shares. It does no I/O and has no async code ([plan §6.2](../../docs/implementation-plan.md#62-layers-and-crates)).

P1.01 creates it nearly empty. P1.02 adds units, geometry, the clock, IDs and errors; P1.03 the frame model.

## How to try it

```bash
cargo run -p tetera-core --example hello
cargo run -p tetera-core --example hello -- --smoke
```

The first prints `Tetera 0.1.0`. The second prints it too, then `smoke: ok`, and exits 0: the gates run every example this way.

To try the debugger, open the repository in VS Code over Remote-WSL, set a breakpoint in `examples/hello.rs`, choose **Debug an example** in the Run and Debug view, enter `hello` when asked, and press F5.
