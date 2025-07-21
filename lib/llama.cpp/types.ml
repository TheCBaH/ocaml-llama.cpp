(** Vocabulary types. *)
module VocabType = struct
  type t =
    | None  (** For models without vocab *)
    | Spm  (** LLaMA tokenizer based on byte-level BPE with byte fallback *)
    | Bpe  (** GPT-2 tokenizer based on byte-level BPE *)
    | Wpm  (** BERT tokenizer based on WordPiece *)
    | Ugm  (** T5 tokenizer based on Unigram *)
    | Rwkv  (** RWKV tokenizer based on greedy tokenization *)
    | Plamo2  (** PLaMo-2 tokenizer based on Aho-Corasick with dynamic programming *)

  let values =
    [ (None, "NONE"); (Spm, "SPM"); (Bpe, "BPE"); (Wpm, "WPM"); (Ugm, "UGM"); (Rwkv, "RWKV"); (Plamo2, "PLAMO2") ]

  let to_string t = List.assoc t values
end

(** RoPE types. *)
module RopeType = struct
  type t = None | Norm | Neox | Mrope | Vision

  let values =
    [
      (None, "NONE");
      (Norm, "NORM");
      (Neox, "NEOX");
      (* GGML_ROPE_TYPE_NEOX *)
      (Mrope, "MROPE");
      (* GGML_ROPE_TYPE_MROPE *)
      (Vision, "VISION");
      (* GGML_ROPE_TYPE_VISION *)
    ]

  let to_string t = List.assoc t values
end

(** Token types (deprecated). *)
module TokenType = struct
  type t = Undefined | Normal | Unknown | Control | UserDefined | Unused | Byte

  let values =
    [
      (Undefined, "UNDEFINED");
      (Normal, "NORMAL");
      (Unknown, "UNKNOWN");
      (Control, "CONTROL");
      (UserDefined, "USER_DEFINED");
      (Unused, "UNUSED");
      (Byte, "BYTE");
    ]

  let to_string t = List.assoc t values
end

(** Token attributes (bit flags). *)
module TokenAttr = struct
  type t =
    | Undefined
    | Unknown
    | Unused
    | Normal
    | Control
    | UserDefined
    | Byte
    | Normalized
    | Lstrip
    | Rstrip
    | SingleWord

  let values =
    [
      (Undefined, "UNDEFINED");
      (Unknown, "UNKNOWN");
      (Unused, "UNUSED");
      (Normal, "NORMAL");
      (Control, "CONTROL");
      (UserDefined, "USER_DEFINED");
      (Byte, "BYTE");
      (Normalized, "NORMALIZED");
      (Lstrip, "LSTRIP");
      (Rstrip, "RSTRIP");
      (SingleWord, "SINGLE_WORD");
    ]

  let to_string t = List.assoc t values
  (* Note: These are bit flags in C, consider using a list or set for multiple attributes *)
end

(** Model file types. *)
module Ftype = struct
  type t =
    | AllF32
    | MostlyF16
    | MostlyQ4_0
    | MostlyQ4_1
    | MostlyQ8_0
    | MostlyQ5_0
    | MostlyQ5_1
    | MostlyQ2K
    | MostlyQ3KS
    | MostlyQ3KM
    | MostlyQ3KL
    | MostlyQ4KS
    | MostlyQ4KM
    | MostlyQ5KS
    | MostlyQ5KM
    | MostlyQ6K
    | MostlyIQ2XXS
    | MostlyIQ2XS
    | MostlyQ2KS
    | MostlyIQ3XS
    | MostlyIQ3XXS
    | MostlyIQ1S
    | MostlyIQ4NL
    | MostlyIQ3S
    | MostlyIQ3M
    | MostlyIQ2S
    | MostlyIQ2M
    | MostlyIQ4XS
    | MostlyIQ1M
    | MostlyBF16
    | MostlyTQ1_0
    | MostlyTQ2_0
    | Guessed

  let values =
    [
      (AllF32, "ALL_F32");
      (MostlyF16, "MOSTLY_F16");
      (MostlyQ4_0, "MOSTLY_Q4_0");
      (MostlyQ4_1, "MOSTLY_Q4_1");
      (MostlyQ8_0, "MOSTLY_Q8_0");
      (MostlyQ5_0, "MOSTLY_Q5_0");
      (MostlyQ5_1, "MOSTLY_Q5_1");
      (MostlyQ2K, "MOSTLY_Q2_K");
      (MostlyQ3KS, "MOSTLY_Q3_K_S");
      (MostlyQ3KM, "MOSTLY_Q3_K_M");
      (MostlyQ3KL, "MOSTLY_Q3_K_L");
      (MostlyQ4KS, "MOSTLY_Q4_K_S");
      (MostlyQ4KM, "MOSTLY_Q4_K_M");
      (MostlyQ5KS, "MOSTLY_Q5_K_S");
      (MostlyQ5KM, "MOSTLY_Q5_K_M");
      (MostlyQ6K, "MOSTLY_Q6_K");
      (MostlyIQ2XXS, "MOSTLY_IQ2_XXS");
      (MostlyIQ2XS, "MOSTLY_IQ2_XS");
      (MostlyQ2KS, "MOSTLY_Q2_K_S");
      (MostlyIQ3XS, "MOSTLY_IQ3_XS");
      (MostlyIQ3XXS, "MOSTLY_IQ3_XXS");
      (MostlyIQ1S, "MOSTLY_IQ1_S");
      (MostlyIQ4NL, "MOSTLY_IQ4_NL");
      (MostlyIQ3S, "MOSTLY_IQ3_S");
      (MostlyIQ3M, "MOSTLY_IQ3_M");
      (MostlyIQ2S, "MOSTLY_IQ2_S");
      (MostlyIQ2M, "MOSTLY_IQ2_M");
      (MostlyIQ4XS, "MOSTLY_IQ4_XS");
      (MostlyIQ1M, "MOSTLY_IQ1_M");
      (MostlyBF16, "MOSTLY_BF16");
      (MostlyTQ1_0, "MOSTLY_TQ1_0");
      (MostlyTQ2_0, "MOSTLY_TQ2_0");
      (Guessed, "GUESSED");
    ]

  let to_string t = List.assoc t values
end

(** RoPE scaling types. *)
module RopeScalingType = struct
  type t = Unspecified | None | Linear | Yarn | Longrope | MaxValue

  let values =
    [
      (Unspecified, "UNSPECIFIED");
      (None, "NONE");
      (Linear, "LINEAR");
      (Yarn, "YARN");
      (Longrope, "LONGROPE");
      (MaxValue, "MAX_VALUE");
    ]

  let to_string t = List.assoc t values
end

(** Pooling types. *)
module PoolingType = struct
  type t = Unspecified | None | Mean | Cls | Last | Rank

  let values =
    [ (Unspecified, "UNSPECIFIED"); (None, "NONE"); (Mean, "MEAN"); (Cls, "CLS"); (Last, "LAST"); (Rank, "RANK") ]

  let to_string t = List.assoc t values
end

(** Attention types. *)
module AttentionType = struct
  type t = Unspecified | Causal | NonCausal

  let values = [ (Unspecified, "UNSPECIFIED"); (Causal, "CAUSAL"); (NonCausal, "NON_CAUSAL") ]
  let to_string t = List.assoc t values
end

(** Model splitting modes. *)
module SplitMode = struct
  type t = None | Layer | Row

  let values = [ (None, "NONE"); (Layer, "LAYER"); (Row, "ROW") ]
  let to_string t = List.assoc t values
end

(** Model key-value override types. *)
module ModelKvOverrideType = struct
  type t = Int | Float | Bool | Str

  let values = [ (Int, "INT"); (Float, "FLOAT"); (Bool, "BOOL"); (Str, "STR") ]
  let to_string t = List.assoc t values
end
