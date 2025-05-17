open Ctypes

module Functions (F : Ctypes.FOREIGN) = struct
  open F
  open Types_generated
  open Types_generated.CPU

  (** [numa_init strategy] initializes NUMA (Non-Uniform Memory Access) support.
      - [strategy] The NUMA strategy to use.*)
  let numa_init = foreign (ns "numa_init") (numa_strategy @-> returning void)

  (** [is_numa ()] checks if NUMA support is enabled.
      - returns True if NUMA is enabled, false otherwise. *)
  let is_numa = foreign (ns "is_numa") (void @-> returning bool)

  (** [new_i32 ctx value] creates a new scalar tensor of type i32.
      - [ctx] The context.
      - [value] The int32 value.
      - returns the new scalar tensor. *)
  let new_i32 = foreign (ns "new_i32") (context @-> int32_t @-> returning tensor)

  (** [new_f32 ctx value] creates a new scalar tensor of type f32.
      - [ctx] The context.
      - [value] The float value.
      - returns The new scalar tensor. *)
  let new_f32 = foreign (ns "new_f32") (context @-> float @-> returning tensor)

  (** [set_i32 tensor value] sets the value of a scalar i32 tensor. Returns the tensor itself.
      - [tensor] The scalar tensor (modified).
      - [value] The int32 value to set.
      - returns The modified tensor. *)
  let set_i32 = foreign (ns "set_i32") (tensor @-> int32_t @-> returning tensor)

  (** [set_f32 tensor value] sets the value of a scalar f32 tensor. Returns the tensor itself.
      - [tensor] The scalar tensor (modified).
      - [value] The float value to set.
      - returns The modified tensor. *)
  let set_f32 = foreign (ns "set_f32") (tensor @-> float @-> returning tensor)

  (** [get_i32_1d tensor i] gets the i32 value at index `i` in a 1D tensor.
      - [tensor] The 1D tensor.
      - [i] The index.
      - returns The int32 value. *)
  let get_i32_1d = foreign (ns "get_i32_1d") (tensor @-> int @-> returning int32_t)

  (** [set_i32_1d tensor i value] sets the i32 value at index `i` in a 1D tensor.
      - [tensor] The 1D tensor.
      - [i] The index.
      - [value] The int32 value to set. *)
  let set_i32_1d = foreign (ns "set_i32_1d") (tensor @-> int @-> int32_t @-> returning void)

  (** [get_i32_nd tensor i0 i1 i2 i3] gets the i32 value at the specified multi-dimensional index.
      - [tensor] The tensor.
      - [i0] Index for dimension 0.
      - [i1] Index for dimension 1.
      - [i2] Index for dimension 2.
      - [i3] Index for dimension 3.
      - returns The int32 value. *)
  let get_i32_nd = foreign (ns "get_i32_nd") (tensor @-> int @-> int @-> int @-> int @-> returning int32_t)

  (** [set_i32_nd tensor i0 i1 i2 i3 value] sets the i32 value at the specified multi-dimensional index.
      - [tensor] The tensor (modified).
      - [i0] Index for dimension 0.
      - [i1] Index for dimension 1.
      - [i2] Index for dimension 2.
      - [i3] Index for dimension 3.
      - [value] The int32 value to set. *)
  let set_i32_nd = foreign (ns "set_i32_nd") (tensor @-> int @-> int @-> int @-> int @-> int32_t @-> returning void)

  (** [get_f32_1d tensor i] gets the float value at index `i` in a 1D tensor.
      - [tensor] The 1D tensor.
      - [i] The index.
      - returns The float value. *)
  let get_f32_1d = foreign (ns "get_f32_1d") (tensor @-> int @-> returning float)

  (** [set_f32_1d tensor i value] sets the float value at index `i` in a 1D tensor.
      - [tensor] The 1D tensor.
      - [i] The index.
      - [value] The float value to set. *)
  let set_f32_1d = foreign (ns "set_f32_1d") (tensor @-> int @-> float @-> returning void)

  (** [get_f32_nd tensor i0 i1 i2 i3] gets the float value at the specified multi-dimensional index.
      - [tensor] The tensor.
      - [i0] Index for dimension 0.
      - [i1] Index for dimension 1.
      - [i2] Index for dimension 2.
      - [i3] Index for dimension 3.
      - returns The float value. *)
  let get_f32_nd = foreign (ns "get_f32_nd") (tensor @-> int @-> int @-> int @-> int @-> returning float)

  (** [set_f32_nd tensor i0 i1 i2 i3 value] sets the float value at the specified multi-dimensional index.
      - [tensor] The tensor (modified).
      - [i0] Index for dimension 0.
      - [i1] Index for dimension 1.
      - [i2] Index for dimension 2.
      - [i3] Index for dimension 3.
      - [value] The float value to set. *)
  let set_f32_nd = foreign (ns "set_f32_nd") (tensor @-> int @-> int @-> int @-> int @-> float @-> returning void)

  (** [threadpool_new params] creates a new threadpool.
      - [params] Threadpool parameters (currently unused, pass NULL).
      - returns Pointer to the new threadpool. *)
  let threadpool_new = foreign (ns "threadpool_new") (ptr void @-> returning (ptr threadpool))

  (** [threadpool_free pool] frees the resources associated with a threadpool.
      - [pool] Pointer to the threadpool to free. *)
  let threadpool_free = foreign (ns "threadpool_free") (ptr threadpool @-> returning void)

  (*
  (** [threadpool_get_n_threads pool] gets the number of threads in the pool.
      - [pool] Pointer to the threadpool.
      - returns Number of threads. *)
  let threadpool_get_n_threads = foreign (ns "threadpool_get_n_threads") (ptr threadpool @-> returning int)
  *)

  (** [threadpool_pause pool] pauses the threads in the threadpool.
      - [pool] Pointer to the threadpool. *)
  let threadpool_pause = foreign (ns "threadpool_pause") (ptr threadpool @-> returning void)

  (** [threadpool_resume pool] resumes the threads in the threadpool.
      - [pool] Pointer to the threadpool. *)
  let threadpool_resume = foreign (ns "threadpool_resume") (ptr threadpool @-> returning void)

  (** [graph_plan graph n_threads pool] creates a computation plan for the graph.
      - [graph] The computation graph.
      - [n_threads] Number of threads to use (overrides pool setting if pool is NULL).
      - [pool] Optional threadpool to use.
      - returns The computation plan. *)
  let graph_plan = foreign (ns "graph_plan") (cgraph @-> int @-> ptr threadpool @-> returning Cplan.t)

  (** [graph_compute graph plan] computes the graph according to the plan.
      - [graph] The computation graph.
      - [plan] Pointer to the computation plan.
      - returns Computation status (`GGML_STATUS_SUCCESS` on success). *)
  let graph_compute = foreign (ns "graph_compute") (cgraph @-> ptr Cplan.t @-> returning status)

  (** [graph_compute_with_ctx ctx graph n_threads] computes the graph using the provided context and number of threads.
      - [ctx] The context (used for backend selection, e.g., CPU).
      - [graph] The computation graph.
      - [n_threads] Number of threads to use.
      - returns Computation status. *)
  let graph_compute_with_ctx = foreign (ns "graph_compute_with_ctx") (context @-> cgraph @-> int @-> returning status)

  (** [cpu_has_sse3 ()] checks if the CPU supports SSE3 instructions.
      - returns 1 if supported, 0 otherwise. *)
  let cpu_has_sse3 = foreign (ns "cpu_has_sse3") (void @-> returning int)

  (** [cpu_has_ssse3 ()] checks if the CPU supports SSSE3 instructions.
      - returns 1 if supported, 0 otherwise. *)
  let cpu_has_ssse3 = foreign (ns "cpu_has_ssse3") (void @-> returning int)

  (** [cpu_has_avx ()] checks if the CPU supports AVX instructions.
      - returns 1 if supported, 0 otherwise. *)
  let cpu_has_avx = foreign (ns "cpu_has_avx") (void @-> returning int)

  (** [cpu_has_avx_vnni ()] checks if the CPU supports AVX VNNI instructions.
      - returns 1 if supported, 0 otherwise. *)
  let cpu_has_avx_vnni = foreign (ns "cpu_has_avx_vnni") (void @-> returning int)

  (** [cpu_has_avx2 ()] checks if the CPU supports AVX2 instructions.
      - returns 1 if supported, 0 otherwise. *)
  let cpu_has_avx2 = foreign (ns "cpu_has_avx2") (void @-> returning int)

  (** [cpu_has_bmi2 ()] checks if the CPU supports BMI2 instructions.
      - returns 1 if supported, 0 otherwise. *)
  let cpu_has_bmi2 = foreign (ns "cpu_has_bmi2") (void @-> returning int)

  (** [cpu_has_f16c ()] checks if the CPU supports F16C (half-precision conversion) instructions.
      - returns 1 if supported, 0 otherwise. *)
  let cpu_has_f16c = foreign (ns "cpu_has_f16c") (void @-> returning int)

  (** [cpu_has_fma ()] checks if the CPU supports FMA (fused multiply-add) instructions.
      - returns 1 if supported, 0 otherwise. *)
  let cpu_has_fma = foreign (ns "cpu_has_fma") (void @-> returning int)

  (** [cpu_has_avx512 ()] checks if the CPU supports AVX512 instructions.
      - returns 1 if supported, 0 otherwise. *)
  let cpu_has_avx512 = foreign (ns "cpu_has_avx512") (void @-> returning int)

  (** [cpu_has_avx512_vbmi ()] checks if the CPU supports AVX512 VBMI instructions.
      - returns 1 if supported, 0 otherwise. *)
  let cpu_has_avx512_vbmi = foreign (ns "cpu_has_avx512_vbmi") (void @-> returning int)

  (** [cpu_has_avx512_vnni ()] checks if the CPU supports AVX512 VNNI instructions.
      - returns 1 if supported, 0 otherwise. *)
  let cpu_has_avx512_vnni = foreign (ns "cpu_has_avx512_vnni") (void @-> returning int)

  (** [cpu_has_avx512_bf16 ()] checks if the CPU supports AVX512 BF16 instructions.
      - returns 1 if supported, 0 otherwise. *)
  let cpu_has_avx512_bf16 = foreign (ns "cpu_has_avx512_bf16") (void @-> returning int)

  (** [cpu_has_amx_int8 ()] checks if the CPU supports AMX INT8 instructions.
      - returns 1 if supported, 0 otherwise. *)
  let cpu_has_amx_int8 = foreign (ns "cpu_has_amx_int8") (void @-> returning int)

  (** [cpu_has_neon ()] checks if the CPU supports NEON instructions (ARM).
      - returns 1 if supported, 0 otherwise. *)
  let cpu_has_neon = foreign (ns "cpu_has_neon") (void @-> returning int)

  (** [cpu_has_arm_fma ()] checks if the CPU supports ARM FMA instructions.
      - returns 1 if supported, 0 otherwise. *)
  let cpu_has_arm_fma = foreign (ns "cpu_has_arm_fma") (void @-> returning int)

  (** [cpu_has_fp16_va ()] checks if the CPU supports FP16 vector arithmetic (ARM).
      - returns 1 if supported, 0 otherwise. *)
  let cpu_has_fp16_va = foreign (ns "cpu_has_fp16_va") (void @-> returning int)

  (** [cpu_has_dotprod ()] checks if the CPU supports Dot Product instructions (ARM).
      - returns 1 if supported, 0 otherwise. *)
  let cpu_has_dotprod = foreign (ns "cpu_has_dotprod") (void @-> returning int)

  (** [cpu_has_matmul_int8 ()] checks if the CPU supports INT8 matrix multiplication instructions (ARM).
      - returns 1 if supported, 0 otherwise. *)
  let cpu_has_matmul_int8 = foreign (ns "cpu_has_matmul_int8") (void @-> returning int)

  (** [cpu_has_sve ()] checks if the CPU supports SVE instructions (ARM).
      - returns 1 if supported, 0 otherwise. *)
  let cpu_has_sve = foreign (ns "cpu_has_sve") (void @-> returning int)

  (** [cpu_get_sve_cnt ()] gets the SVE vector length.
      - returns SVE vector length in bits. *)
  let cpu_get_sve_cnt = foreign (ns "cpu_get_sve_cnt") (void @-> returning int)

  (** [cpu_has_sme ()] checks if the CPU supports SME instructions (ARM).
      - returns 1 if supported, 0 otherwise. *)
  let cpu_has_sme = foreign (ns "cpu_has_sme") (void @-> returning int)

  (** [cpu_has_riscv_v ()] checks if the CPU supports RISC-V Vector extension.
      - returns 1 if supported, 0 otherwise. *)
  let cpu_has_riscv_v = foreign (ns "cpu_has_riscv_v") (void @-> returning int)

  (** [cpu_has_vsx ()] checks if the CPU supports VSX instructions (PowerPC).
      - returns 1 if supported, 0 otherwise. *)
  let cpu_has_vsx = foreign (ns "cpu_has_vsx") (void @-> returning int)

  (** [cpu_has_vxe ()] checks if the CPU supports VXE instructions (PowerPC).
      - returns 1 if supported, 0 otherwise. *)
  let cpu_has_vxe = foreign (ns "cpu_has_vxe") (void @-> returning int)

  (** [cpu_has_wasm_simd ()] checks if the environment supports WASM SIMD.
      - returns 1 if supported, 0 otherwise. *)
  let cpu_has_wasm_simd = foreign (ns "cpu_has_wasm_simd") (void @-> returning int)

  (** [cpu_has_llamafile ()] checks if running within llamafile environment.
      - returns 1 if true, 0 otherwise. *)
  let cpu_has_llamafile = foreign (ns "cpu_has_llamafile") (void @-> returning int)

  (** [get_type_traits_cpu typ] gets the CPU-specific type traits for a given type.
      - [typ] The ggml type.
      - returns Pointer to the constant type traits structure. *)
  let get_type_traits_cpu = foreign (ns "get_type_traits_cpu") (typ @-> returning (ptr @@ const TypeTraitsCpu.t))

  (** [cpu_init ()] initializes CPU-specific features (e.g., detects capabilities). *)
  let cpu_init = foreign (ns "cpu_init") (void @-> returning void)

  (** [backend_cpu_init ()] initializes the CPU backend.
      - returns The CPU backend handle. *)
  let backend_cpu_init = foreign (ns "backend_cpu_init") (void @-> returning Backend.backend_t)

  (** [backend_is_cpu backend] checks if the given backend is the CPU backend.
      - [backend] The backend handle.
      - returns True if it's the CPU backend, false otherwise. *)
  let backend_is_cpu = foreign (ns "backend_is_cpu") (Backend.backend_t @-> returning bool)

  (** [backend_cpu_set_n_threads backend n_threads] sets the number of threads for the CPU backend.
      - [backend] The CPU backend handle.
      - [n_threads] The desired number of threads. *)
  let backend_cpu_set_n_threads = foreign (ns "backend_cpu_set_n_threads") (Backend.backend_t @-> int @-> returning void)

  (** [backend_cpu_set_threadpool backend pool] sets a custom threadpool for the CPU backend.
      - [backend] The CPU backend handle.
      - [pool] Pointer to the threadpool. *)
  let backend_cpu_set_threadpool =
    foreign (ns "backend_cpu_set_threadpool") (Backend.backend_t @-> ptr threadpool @-> returning void)

  (** [backend_cpu_set_abort_callback backend callback data] sets an abort callback for the CPU backend.
      - [backend] The CPU backend handle.
      - [callback] The callback function.
      - [data] User data to pass to the callback. *)
  let backend_cpu_set_abort_callback =
    foreign (ns "backend_cpu_set_abort_callback") (Backend.backend_t @-> abort_callback @-> ptr void @-> returning void)

  (** [backend_cpu_reg ()] gets the registration information for the CPU backend.
      - returns The backend registration structure. *)
  let backend_cpu_reg = foreign (ns "backend_cpu_reg") (void @-> returning Backend.reg_t)

  (** [cpu_fp32_to_fp16 src dst n] converts an array of float32 values to float16.
      - [src] Pointer to the source float32 array.
      - [dst] Pointer to the destination float16 array.
      - [n] Number of elements to convert. *)
  let cpu_fp32_to_fp16 = foreign (ns "cpu_fp32_to_fp16") (ptr float @-> ptr fp16_t @-> int64_t @-> returning void)

  (** [cpu_fp16_to_fp32 src dst n] converts an array of float16 values to float32.
      - [src] Pointer to the source float16 array.
      - [dst] Pointer to the destination float32 array.
      - [n] Number of elements to convert. *)
  let cpu_fp16_to_fp32 = foreign (ns "cpu_fp16_to_fp32") (ptr fp16_t @-> ptr float @-> int64_t @-> returning void)

  (** [cpu_fp32_to_bf16 src dst n] converts an array of float32 values to bfloat16.
      - [src] Pointer to the source float32 array.
      - [dst] Pointer to the destination bfloat16 array.
      - [n] Number of elements to convert. *)
  let cpu_fp32_to_bf16 = foreign (ns "cpu_fp32_to_bf16") (ptr float @-> ptr bf16_t @-> int64_t @-> returning void)

  (** [cpu_bf16_to_fp32 src dst n] converts an array of bfloat16 values to float32.
      - [src] Pointer to the source bfloat16 array.
      - [dst] Pointer to the destination float32 array.
      - [n] Number of elements to convert. *)
  let cpu_bf16_to_fp32 = foreign (ns "cpu_bf16_to_fp32") (ptr bf16_t @-> ptr float @-> int64_t @-> returning void)
end
