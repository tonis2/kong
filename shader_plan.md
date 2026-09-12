# shader_plan.md

Making `lib/shady.c3l` the shader compiler for `../three.c3`, with the engine's
shader ABI sourced from C3 rather than from hand-written shader text.

Working checklist: tick items as they land. `LANGUAGE.md` stays the language
specification; this file is the task list.

---

## Strategy

The old flow compiles hand-written Slang templates that the host strings
together with text markers and `#define`/`#ifdef`. The new flow removes the
templating layer entirely:

1. **The C3 structs are the ABI.** `DrawRecord`, `Instance`, `FrameBlock`,
   `MeshPush`, `SkinPush`, `ClusterPush` already are the wire format. Generate
   the shader-side struct declarations from them so layout cannot drift, and
   drop the `$assert` / `check_push_block` / reflection cross-checks that exist
   only to catch that drift.
2. **The host generates shady source; there are no `.slang` templates.** A
   `ShaderDef` in C3 describes resources, push block, feature knobs and the
   hooks where agent bodies go. The generator emits the final program.
3. **No preprocessor.** With generated source there is no `#define`, no
   `#ifdef`, and no marker splicing. Shady needs no preprocessing at all beyond
   a source map for diagnostics on agent bodies.
4. **Variants are chosen at the right level** — see the table below.
5. **Specialization constants** carry fixed value knobs so one module serves
   several pipelines without recompiling.

```
three.c3 (C3)                              shady.c3l
  DrawRecord / Instance / FrameBlock
        │ single source of layout
        ▼
  ShaderDef ── emit structs / resources ──►  source string ──► SPIR-V
   ├ resources (set, binding)                        │
   ├ push block (from C3 struct)                     ▼
   ├ features ─► spec constants               VkShaderModule
   ├ variant ──► module select                       │
   └ hooks ◄── agent body text                       ▼
                                             pipeline cache (+ spec values)
```

## The contract that still has to hold

| Contract | Notes under the new plan |
|---|---|
| Entry-point names `vertexMain` / `fragmentMain` / `computeMain` | preserved by the generator |
| Descriptor layout (set 0 frame samplers, set 1 bindless array) | generator knows it; no reflection round-trip needed |
| Push-constant byte offsets | guaranteed by construction from the C3 struct |
| Buffer struct strides (`Draw`, `Instance`, `FrameBlock`, `SkinnedVertex`) | guaranteed by construction |
| Column-major matrices, `mul(M, v)` | unchanged |
| Live agent bodies with line-accurate diagnostics | kept — the reason runtime generation stays |
| Empty vertex input state; geometry via device addresses | unchanged |

## Variant policy

| Axis | Example | Mechanism |
|---|---|---|
| Interface / whole-program change | bake, lightmap bake, shadow cut-out, vertex-body-present | **separate generated module** |
| Algorithm with different resources | area-reference vs LTC | **separate generated module** |
| Fixed numeric / bool knob | PCF taps, shadow filter mode, AO on where the binding stays bound, tonemap operator | **specialization constant** |
| Per-material / per-draw | `FLAG_HAS_*`, `FLAG_STOCHASTIC`, `environment.x`, `lighting.w` | **push constant** (as today) |

Specialization constants cannot add or remove descriptors, change varyings or
entry points, or change stage interfaces. They belong only in the third row.

## What this removes from shady's required surface

| Area | Old plan | Under the new plan |
|---|---|---|
| Preprocessor (`#define`/`#if`/`#undef`) | required, large | **not needed** |
| Marker splicing | shady or host | host only, and only to place agent bodies |
| Semantic strings (`SV_Position`, `TEXCOORDn`… ) | required | **not needed** — the generator emits `@position`/`@builtin`/`@flat` directly |
| Slang-style reflection to drive the push block | required | **not needed** — layout comes from C3; compiler only reports descriptors |
| `#line` | required | keep, as a source map for agent bodies |
| Arrays, `break`/`continue`, operators, pointers, atomics, descriptor arrays, samplers, scalar layout, spec constants | required | **still required** |

## Decisions to settle first

- [x] **C3 → shader declarations.** Confirmed on c3c 0.8.3: `$Type::members`
      exposes each member's `name`, `offset` and `size` at compile time, and
      `$Type::size` the total. Already relied on at
      `three.c3/src/gpu/pipeline.c3:698` and `three.c3/test/shader_test.c3:808`.
      The generator can walk it to emit shader structs in declaration order.
- [x] **Field types for the walk.** `$member.type` is exposed and used by the
      stdlib (`lib/std/encoding/json_marshal.c3:45`, `:204`); a `typeid` carries
      `.kind` (e.g. `ARRAY`, `POINTER`), `.inner` and `.size`
      (`lib/std/io/formatter_private.c3:792`, `:913`). A generator can recurse
      C3 types into shady types with no manifest.
- [x] **Type → shady spelling.** `src/shader/abi.c3`, as Phase 7 above: the
      mapping is by `$Type::kind`, arrays recurse `$Type::inner`, and the
      extent is `$Type::len` read into a macro variable (`$Type::len` is
      compile-time only, so it cannot be read at the runtime use site). The
      pointee of an address comes from the C3 pointer type, so no manifest -
      which is what typing those fields as pointers in `ec2a867` bought.
- [x] **Scalar layout.** Adopted: `compile(..., scalar_layout: true)` packs like
      C3 rather than std140/std430. Pinned by `test/layout_test.c3` against the
      `DrawRecord` / `MeshPush` / `SkinnedVertex` / post-push offsets. The
      generator that uses it is Phase 7.
- [x] **Agent-facing API.** Keep the surface. Methods and overloading are in
      (`lib/shady.c3l/shady/codegen.c3`): methods are declared C3-style, `fn R
      Type.name(&self, ...)`, and `&self` arrives as a real pointer into the
      caller's storage, so `Map.Sample(uv)` works and writes through `&self`
      reach an addressable receiver. Overloads resolve by arity and argument
      type. `out`/`inout` (for `displace(inout Vertex)`) is in on the same
      by-pointer mechanism, and arrays, casts, ternaries and the operator set a
      ported body needs are in too.
- [x] **Buffer device addresses.** Kept: BDA is the model, and three.c3's wire
      structs now spell their addresses as real pointers (`ec2a867`). Shady's
      pointer surface covers the streams - `float3*`, `uint4*`, `float4x4*`, and
      pointers to wire structs.
- [ ] **Atomics.** Implement `InterlockedAdd`, or restructure the cluster
      overflow path to avoid it.
- [ ] **Comparison samplers.** Add a comparison sampler type, or replace the
      shadow PCF with a raw-depth manual compare.
- [ ] **Combined samplers.** Add a combined opaque type so binding and
      reflection say `COMBINED_IMAGE_SAMPLER`, matching the C3 side.
- [ ] **Spec-constant list.** Enumerate which knobs become spec constants, with
      ids, types and defaults (start: shadow PCF/filter mode, AO enable,
      tonemap/post mode).
- [ ] **Optimizer.** None. Drivers optimize; three.c3 compiles at `-O0` today.

---

## Phase 0 — Conformance harness

Every later phase is guesswork without this.

- [x] Test tool: `lib/shady.c3l/test/conformance.c3` compiles each corpus source
      with shady, writes `build/shady/conformance/<name>.spv`, and runs
      `spirv-val --target-env vulkan1.3` (not generic spirv-val — the Vulkan
      env is what catches `BuiltIn Position` in a fragment shader). A missing
      `spirv-val` skips with one message rather than failing. Sources are
      `$embed`ed, so it runs the same standalone and from kong.
- [x] Minimal smoke test with one uniform, one texture and one entry point:
      `lib/shady.c3l/test/conformance/triangle.shady`. Verified: the suite is
      green in `lib/shady.c3l` and in kong, and every corpus module validates
      by hand with `spirv-val --target-env vulkan1.3`.
- [x] Corpus of language features: nine modules under
      `lib/shady.c3l/test/conformance/` (`triangle`, `pushconstant`, `bda`,
      `multi_entry`, `compute`, `functions`, `operators`, `arrays`,
      `builtins`), each compiled and validated.
- [ ] Capture expected artifacts from the current Slang build: entry points,
      descriptor `(name, set, binding, count, type)`, push field
      `(name, offset, size)`, push size, struct strides. **Blocked on
      three.c3** — kong has no Slang dependency; add a dumper in three.c3 when
      integration starts, then check the fixtures in here.
- [ ] Add the nine three.c3 templates to the corpus as each is ported in
      Phase 9. Scalar layout (Phase 2) is the last language gap they need.

**Done when**: the harness is green, and the Slang fixtures exist for every
three.c3 corpus shader (the second item is the only part not yet startable in
kong).

## Phase 1 — shady language core

- [x] Non-entry-point functions (helpers), including forward references: a call
      may name a function declared later.
- [x] Methods and overloading, C3-style: `fn R Type.name(&self, ...)`, with
      `&self` passed by pointer, plus overload resolution by arity and argument
      type. Ambiguity and recursion are errors.
- [x] `out` / `inout` parameters, passed by pointer like `&self`. The
      `displace(inout Vertex)` agent API.
- [x] `break` and `continue`.
- [x] Ternary `?:`, lowered to a real branch so only the taken arm runs.
- [x] Prefix/postfix `++` / `--`.
- [x] Compound bitwise assignment `|=`, `^=`, `&=`, `<<=`, `>>=`.
- [x] C-style casts `(float)x`, `(uint3)v`, `(float3x3)m`, including matrix
      narrowing. Only builtin type names cast, so `(x) - y` stays arithmetic.
- [x] Arrays: locals and struct members, `T[n]` and `T name[n]`, initialiser
      lists, dynamic indexing, arrays of structs, arrays reached through an
      `@address` pointer. An array parameter is refused with a pointer
      suggestion.
- [x] `mul(M, v)` / `mul(M, M)` matched to the `*` convention.
- [x] Builtin gaps: `lerp`, `rsqrt`, `asuint`/`asint`/`asfloat`, `all`, `any`,
      `select`. Scalar arguments to vector GLSL instructions are broadcast.
- [x] `[unroll]` accepted and ignored.
- [x] Mutable module-level `static`: not needed. The generator inlines
      `static const` values and passes state as values or parameters, so shady
      stays without mutable globals.

**Done when**: the generated engine library sources (surface/lighting maths)
parse and validate, and unit tests cover arrays, loops and operators. The unit
tests and the nine-module conformance corpus are green; the library sources
themselves are ported in Phase 9.

## Phase 2 — Scalar layout mode

- [x] Layout mode for `@uniform` / `@pushconstant` / `@address` that packs like
      C3: `float3` is 12 bytes aligned 4; structs do not round to 16; matrices
      use the column size as stride.
- [x] Selectable per compile (`scalar_layout: true`), so existing std140/std430
      tests keep passing.
- [x] Offset tests pinning `Draw.uv_transform == 108`, `texture_index == 288`,
      `SkinnedVertex` stride 24, `MeshPush` 24 and `PostPush` 24.
      `test/layout_test.c3` reads the offsets back off the emitted
      `OpMemberDecorate ... Offset`; `test/conformance/scalar_layout.shady` is
      validated with `--scalar-block-layout`.
- [x] Nested structs and arrays of structs are laid out and decorated by the
      enclosing block, recursively. A struct shared by two blocks that lay it
      out differently is refused rather than silently resolved (it is one type
      with one layout).

**Done when**: generated structs and the C3 structs agree on every offset and
stride by construction and by test. The shady side is pinned; the C3 side joins
in Phase 7, where the generator emits from the wire structs themselves.

## Phase 3 — Pointers, writes, atomics

- [x] Pointees beyond `@address` structs: `float3*`, `uint*`, `uint4*`,
      `float*`, `float4x4*` and the rest of the flat streams.
- [x] Writes through a pointer and through `p[i]`, with the `Aligned` operand.
- [x] Pointer reassignment, pointer parameters, and choosing between two
      addresses with a ternary. There is no `null` literal and the shaders do
      not test a pointer for it - the "is this live" bit is a packed `ulong`
      count, not an address.
- [x] Scalar-layout strides for pointer indexing. Fixing this found a real bug:
      the stride used the element's bare size, so a std430 `float3*` stepped 12
      bytes instead of 16. Both the arithmetic and the `ArrayStride` now use
      the strided size.
- [x] The explicit 64-bit arithmetic lowering is unchanged; `OpPtrAccessChain`
      is still avoided for the RADV reason.
- [x] `InterlockedAdd` on device memory (device scope, relaxed - Vulkan forbids
      `SequentiallyConsistent`).

Also needed and added on the way: `scalar * matrix` and matrix `+`/`-`
(`blend_joints` spends both), and two structured-control-flow fixes where a
condition was loaded between `OpSelectionMerge` and its branch, in `if` and in
the ternary.

**Done when**: skin + cluster compute compile, validate and read back written
values in a dispatch smoke test. The shapes compile and validate
(`test/conformance/pointers.shady`, `test/pointer_test.c3`); the read-back needs
a device and lands with the engine integration in Phase 9.

## Phase 4 — Descriptors and samplers

- [x] Runtime-sized opaque arrays as module-level resources
      (`Sampler2D three_textures[]`), with `RuntimeDescriptorArray` and
      `SPV_EXT_descriptor_indexing`. A runtime-sized array anywhere else is
      refused.
- [x] `NonUniformResourceIndex(index)` - the index is decorated and yielded
      unchanged, with `ShaderNonUniform`.
- [x] Combined image sampler type: `Sampler2D` is one `OpTypeSampledImage`
      descriptor that samples directly, alongside the separate
      `texture2d` + `sampler` pair.
- [x] Comparison sampler type: `Sampler2DShadow` (`Depth` operand 1), sampled
      with `SampleCmpLevelZero` → `OpImageSampleDrefExplicitLod`, yielding a
      float.
- [x] Sampling variants `SampleGrad` and `SampleBias` beside
      `Sample`/`SampleLevel`. The Slang spellings are accepted.
- [x] Two sets in one module: the bindless table at set 1, the material's own
      samplers at set 0, decorated and validated.

**Done when**: the bindless `Map` path compiles and descriptor output matches
the fixtures. `test/conformance/bindless.shady` is the `surface.slang` shape -
runtime table, `NonUniformResourceIndex`, shadow compare, all four variants,
two sets - and validates. Descriptor *output* (a list for the host) is not
emitted yet; the generator knows the bindings from the manifest, so Phase 7
decides whether it is needed.

## Phase 5 — Specialization constants

- [x] Syntax: `const T NAME @spec(id) = literal;`. Scalars only, literal
      initialisers only. Without `@spec` it is an ordinary `OpConstant`, which
      makes module-level constants work as a side effect.
- [x] `OpSpecConstant` / `OpSpecConstantTrue` / `OpSpecConstantFalse` +
      `SpecId`, with `OpName` so the ids can be read out of a disassembly.
- [x] Duplicate spec ids are an error rather than one silently winning.
- [x] `OpSpecConstantOp` is not needed: a spec constant used as a loop bound or
      in a comparison is an ordinary operand of the loop.
- [x] Decorated so `spirv-val --target-env vulkan1.3` accepts it
      (`test/conformance/spec.shady`).
- [x] `LANGUAGE.md` §3.4 and the attribute table.

**Done when**: a boolean spec constant removes a branch, and two pipelines made
from one module behave differently. The module side is pinned - a bool gates a
pass, a uint bounds a loop, a float scales a kernel, each named and `SpecId`-
decorated - and validates. Observing the two pipelines is a device test, which
lands with the engine integration (Phase 9/11). Reflection is not emitted: the
generator writes the `@spec` constants from the `ShaderDef`, so it already knows
name, id, type and default, and the `OpName` is there for tooling.

## Phase 6 — Diagnostics and source map

- [x] `Diagnostic` carries the region name, the line and column in that
      region's numbering, the offending source line and the caret span.
      `render` prints heading + line + caret the way a compiler does;
      `to_string` gives the bare `material:3:12: message` heading.
- [x] `#line <n> "<name>"` directives, read by the lexer and recorded in a
      `LineMap`. Every pass keeps physical positions and `compile` maps once,
      because the map's lifetime is the compile's and not a pass's.
- [x] A lexer fault now carries its message and position to the caller. It used
      to arrive as an empty diagnostic: `tokenize` propagated the fault and
      dropped the lexer that held them.
- [x] Warnings beside the fault rather than in it:
      `compile(..., warnings: &list)` collects them whether the compile
      succeeded or not. First one: an unread `@spec` constant, which is a host
      knob wired to nothing.
- [x] `LANGUAGE.md` §1.1, README §diagnostics, and the renderer's exact shape
      pinned in `test/diagnostic_test.c3`.

**Done when**: an undefined identifier in a material body reports
`material:3:12` in the current shape. Pinned by
`a_body_is_reported_by_its_own_line_and_name`, with the rendered block -
heading, source line and caret - in
`the_rendered_diagnostic_shows_the_line_and_the_caret`.

## Phase 7 — ABI single-source (three.c3 side)

- [x] **The wire structs carry real pointers for their device addresses.**
      `DrawRecord.positions` is `Vec3*`, not a `ulong` holding an address: a
      `ulong` does not say what it points at, and the shader declaration needs
      the pointee. Layout is unchanged - every `$assert` size and offset holds -
      and the compiler found every assignment site. three.c3 `ec2a867`.
- [x] **`src/shader/abi.c3`** walks a struct with `$Type::members` and emits the
      shady declaration in declaration order (`append_shady_struct`), plus the
      whole wire format in dependency order (`append_shader_abi`). Mapping:
      scalars keep their names; a vector, and a 2..4 array of scalars, is
      `float4`-style; `Matrix4f` is `float4x4`; a struct keeps its own name; a
      pointer appends `*`. An unmapped type is a `$error` at the field that
      caused it.
- [x] **`test/abi_test.c3` holds the emitter to the running format**: every
      declaration is cut out of the shader that declares it today and compared
      with comments and whitespace removed. `MeshPush`/`ShadowPush` compare
      against their `PushData`; the only folded difference is
      `Draw`/`DrawRecord`, which the port renames.
- [ ] **Emit resources from a manifest** rather than from hand-written
      bindings. Wants the `ShaderDef` of Phase 8 to carry it.
- [ ] **Delete the duplicated `struct Draw` / `Instance` / `FrameBlock`
      declarations from the shader text** - the moment generated source is what
      gets compiled, which is Phase 8.
- [ ] **Keep `check_push_block` as a test, not a runtime gate.** The runtime
      gate exists because the declaration is written twice; there is one after
      the deletion, which is Phase 8 too.

**Done when**: one edit to a C3 wire struct flows to both sides with no manual
shader edit. The generation half is in and pinned; the deletion half waits on
Phase 8, because nothing compiles the generated source yet.

## Phase 8 — ShaderDef generator (three.c3 side)

- [x] **`src/shader/def.c3`**: `ShaderDef` (name, stage IO, resources, features,
      stages) plus the emitters. `append_shader_def(def, $Type)` writes the whole
      program: wire structs (`abi.c3`), the push block from the C3 struct, the
      stage IO structs, the `@spec` constants, the resources with their
      `@set`/`@binding`, and the entry points. One module, no markers, no
      `#define`, no `#ifdef` - pinned by `test/shader_def_test.c3`.
- [x] Entry points carry the stage attribute and the interface: a vertex/fragment
      signature is generated around the body, a compute one carries
      `@threads(x, y, z)`, and stage IO members carry
      `@position`/`@builtin(name)`/`@flat`.
- [x] Features are specialization constants, per the variant policy - one module
      serves every value, so nothing in the interface changes with them.
- [x] Bodies are text injected once, bracketed by `#line 1 "<body>"` and a
      counted restore to `three.<name>` - the same source-map shape Phase 6
      built, now with the count done by the generator instead of by a template.
- [x] **Verified against the compiler**: three.c3's suite writes the generated
      sky module to `build/shaderdef/sky.shady`; kong's
      `test/generated_shader_test.c3` compiles it with shady and runs
      `spirv-val --target-env vulkan1.3`.
- [ ] **Delete `src/shader/assemble.c3`**, the marker constants in `load.c3` and
      the `#define`/`#ifdef` machinery. **Waits for Phase 9**: the engine still
      compiles through the old path, so deleting it now deletes the renderer. The
      new path has none of it, which is what the tests above say.

**Done when**: a ShaderDef produces a complete module with no markers, no macros
and no `#ifdef`. Met for the shape tested: a generated module compiles, validates,
and contains none of the three. The deletion is the last item of Phase 9.

**A finding worth keeping**: shady refuses a bare builtin as an entry-point
parameter - "an entry point parameter must be a struct" - so a vertex stage's
`SV_VertexID` is a one-member IO struct on both sides. The generator would have
had to do it anyway; the compiler said so at the first generated module rather
than after the engine had been ported.

## Phase 9 — Convert the engine shaders

One at a time, each ending in a screenshot parity check against the current
Slang build.

**The first one is checked in kong's renderer, not in three.c3.** Shady is
kong's dependency and not three.c3's, so a ported body can be generated by
three.c3, compiled and *drawn* in kong's Vulkan renderer long before the engine
switches over - which is the point: a shader that is wrong is found in a
hundred-line renderer rather than in the middle of a rewrite. The bridge is
already in place on both sides: three.c3 writes `build/shaderdef/*.shady`, and
`kong/test/generated_shader_test.c3` compiles and validates them.

- [ ] sky
- [ ] skin (compute)
- [ ] cluster (compute)
- [ ] shadow (+ cut-out variant)
- [ ] mesh
- [ ] material (+ bake, lightmap, area-reference variants)
- [ ] post
- [ ] surface / lighting shared library → a shady source module

**Done when**: no `.slang` file remains for the engine, and examples render
identically.

## Phase 10 — Agent body path

- [ ] Material uniforms without `#define`: emit typed access the body can read
      (including the spilled-table case) at generate time.
- [ ] Vertex body (`displace`) path.
- [ ] Post body path.
- [ ] `#line`-equivalent source mapping for each body.
- [ ] Update examples and `docs/` for any API that changed.

**Done when**: `examples/*.js` that attach shader bodies run and report errors
on the right line.

## Phase 11 — Host integration

- [ ] shady-backed `compile` returning the existing `ShaderModule` shape.
- [ ] Replace the Slang argument list in `src/shader/compile.c3`.
- [ ] Pipeline cache key includes spec-constant values (the main trap: two
      variants silently sharing one pipeline).
- [ ] Drop `lib/slang.c3l` from `project.json` and the packaging/setup scripts.

**Done when**: three.c3 builds and runs with no Slang dependency and the full
test suite passes (`c3c test --trust=full`).

## Risks

- **Scalar layout silently renders garbage on mismatch.** Pin with offset
  tests, never by eye.
- **Pointer writes + physical addressing** have driver pitfalls; keep the
  explicit arithmetic lowering and validate against `--target-env vulkan1.3`.
- **Spec-constant keying.** A pipeline cache that ignores spec values looks
  correct and renders the wrong variant.
- **Scope.** Phases 1–5 are the bulk of the work. The payoff is one shader
  language and one ABI across kong and three.c3; the alternative — a minimal
  shady subset plus hand-rewritten templates — moves the cost into the engine
  and keeps the duplication.

## Non-goals

- `#include`, file resolution, offline `.spv`.
- A preprocessor, generics, interfaces, raytracing, mesh shaders.
- An optimizer.
- Supporting arbitrary Slang input.

## Already in place — do not rebuild

- In-process SPIR-V emission and module assembly (`module.c3`, `types.c3`).
- Multiple entry points per module, named per function with override.
- `PhysicalStorageBuffer64` device addresses, `@address`/`@pushconstant`, the
  explicit pointer lowering.
- std140/std430 with `Offset`/`MatrixStride` decorations.
- Structured control flow for `if`/`while`/`for`, with `break`/`continue`.
- Most GLSL.std.450 builtins and vector/matrix arithmetic.
