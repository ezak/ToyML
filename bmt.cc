/*
 * Created by izak on 10/6/26.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */


// This program marks the first step in my journy to understand the transformer
// This code is created by gemini and copied by hand then verified by gemini for typo and copy pested directly

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

/*
 * Model Hyperparameters
 */
struct ModelParams
{
  int32_t n_vocab         = 151936; // Vocabulary size
  int32_t n_embed         = 896;    // Model dimension
  int32_t n_head          = 14;     // Query head
  int32_t h_head_kv       = 2;      // Key/Value head (GQA)
  int32_t n_layer         = 1;      // Single layer for minimalist demonstration
  int32_t n_rot           = 64;     // RoPE dimension per head (head_dim)
  int32_t n_ff            = 4864;   // SwiGLU intermediate dimension
  int32_t max_seq         = 512;    // Maximum KV cache context length
  float   norm_eps        = 1e-6f;
  float   rope_freq_base  = 10000.0f;
  float   rope_freq_scale = 1.0f;
};

/*
 * Container holding raw tensor pointers for model weights
 */
struct ModelWeights
{
  struct ggml_tensor *token_embed;
  struct ggml_tensor *attn_norm;
  struct ggml_tensor *wq;
  struct ggml_tensor *wk;
  struct ggml_tensor *wv;
  struct ggml_tensor *wo;
  struct ggml_tensor *ffn_norm;
  struct ggml_tensor *w_gate;
  struct ggml_tensor *w_up;
  struct ggml_tensor *w_down;
  struct ggml_tensor *output_norm;
  struct ggml_tensor *output_w;
};

/*
 * Container holding explicit KV cache tensors
 * Shape: [head_dim, h_head_kv, max_seq]
 */
struct KVCache
{
  struct ggml_tensor *k;
  struct ggml_tensor *v;
};

/*
 * Graph construction (Node by Node forward pass)
 */
struct ggml_cgraph *
build_transformer_graph (
    struct ggml_context *context,
    const ModelParams &  params,
    ModelWeights &       weights,
    KVCache &            kv,
    struct ggml_tensor * inp_tokens,
    struct ggml_tensor * inp_pos,
    int                  n_past,
    struct ggml_tensor **out_logits
    )
{
  struct ggml_cgraph *graph = ggml_new_graph (context);

  // 1. Token Embedding Lookup
  struct ggml_tensor *cur = ggml_get_rows (context, weights.token_embed, inp_tokens);

  // 2. Input RMSNorm
  struct ggml_tensor *residual = cur;
  cur                          = ggml_rms_norm (context, cur, params.norm_eps);
  cur                          = ggml_mul (context, cur, weights.attn_norm);

  // 3. Q, K, V Projections
  struct ggml_tensor *q = ggml_mul_mat (context, weights.wq, cur);
  struct ggml_tensor *k = ggml_mul_mat (context, weights.wk, cur);
  struct ggml_tensor *v = ggml_mul_mat (context, weights.wv, cur);

  int head_dim = params.n_embed / params.n_head;

  // Reshape Q, K, V for multi-head processing: [head_dim, n_head, n_tokens=1]
  // ne[0] = head_dim, ne[1] = n_head, ne[2] = n_tokens (matches inp_pos->ne[0])
  q = ggml_reshape_3d (context, q, head_dim, params.n_head, 1);
  k = ggml_reshape_3d (context, k, head_dim, params.h_head_kv, 1);
  v = ggml_reshape_3d (context, v, head_dim, params.h_head_kv, 1);

  // 4. Apply Rotary Position Encoding (RoPE)
  q = ggml_rope_ext (context, q, inp_pos, nullptr, params.n_rot, 0, 0, params.rope_freq_base, params.rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
  k = ggml_rope_ext (context, k, inp_pos, nullptr, params.n_rot, 0, 0, params.rope_freq_base, params.rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);

  // 5. Manual Population of KV Cache
  // kv.k shape: [head_dim, h_head_kv, max_seq] -> sequence index is along nb[2]
  size_t k_offset = n_past * kv.k->nb[2];
  size_t v_offset = n_past * kv.v->nb[2];

  struct ggml_tensor *k_cache_view = ggml_view_3d (
      context, kv.k, head_dim, params.h_head_kv, 1,
      kv.k->nb[1], kv.k->nb[2], k_offset
      );
  struct ggml_tensor *v_cache_view = ggml_view_3d (
      context, kv.v, head_dim, params.h_head_kv, 1,
      kv.v->nb[1], kv.v->nb[2], v_offset
      );

  ggml_build_forward_expand (graph, ggml_cpy (context, k, k_cache_view));
  ggml_build_forward_expand (graph, ggml_cpy (context, v, v_cache_view));

  // 6. Active KV Views & Scaled Dot Product Flash Attention
  struct ggml_tensor *k_active = ggml_view_3d (
      context, kv.k, head_dim, params.h_head_kv, n_past + 1,
      kv.k->nb[1], kv.k->nb[2], 0
      );
  struct ggml_tensor *v_active = ggml_view_3d (
      context, kv.v, head_dim, params.h_head_kv, n_past + 1,
      kv.v->nb[1], kv.v->nb[2], 0
      );

  float               scale    = 1.0f / sqrtf (static_cast<float> (head_dim));
  struct ggml_tensor *attn_out = ggml_flash_attn_ext (
      context, q, k_active, v_active, nullptr, scale, 0.0f, 0.0f
      );

  // Reshape back to [n_embed, 1] and apply output projection
  attn_out = ggml_reshape_2d (context, attn_out, params.n_embed, 1);
  cur      = ggml_mul_mat (context, weights.wo, attn_out);
  cur      = ggml_add (context, cur, residual);

  // 7. Feed-Forward Network (SwiGLU MLP)
  residual = cur;
  cur      = ggml_rms_norm (context, cur, params.norm_eps);
  cur      = ggml_mul (context, cur, weights.ffn_norm);

  struct ggml_tensor *w_g = ggml_mul_mat (context, weights.w_gate, cur);
  struct ggml_tensor *w_u = ggml_mul_mat (context, weights.w_up, cur);

  cur = ggml_mul (context, ggml_silu (context, w_g), w_u);
  cur = ggml_mul_mat (context, weights.w_down, cur);
  cur = ggml_add (context, cur, residual);

  // 8. LM Head Output
  cur = ggml_rms_norm (context, cur, params.norm_eps);
  cur = ggml_mul (context, cur, weights.output_norm);

  struct ggml_tensor *logits = ggml_mul_mat (context, weights.output_w, cur);
  ggml_build_forward_expand (graph, logits);

  if (out_logits)
    {
      *out_logits = logits;
    }

  return graph;
}

/*
 * Helper to fill tensor buffer with mock standard normal floats
 */
void
initialize_mock_weights (struct ggml_tensor *t)
{
  std::vector<float> data (ggml_nelements (t));
  for (size_t i = 0; i < data.size (); ++i)
    {
      data[i] = (static_cast<float> (rand ()) / RAND_MAX - 0.5f) * 0.02f;
    }
  ggml_backend_tensor_set (t, data.data (), 0, ggml_nbytes (t));
}

/*
 * Main execution pipeline
 */
int
main ()
{
  ggml_time_init ();
  ModelParams params;

  // Initialize Primary GPU Backend
  ggml_backend_t backend_gpu = ggml_backend_init_best ();

  // Initialize Mandatory CPU Fallback Backend
  ggml_backend_t backend_cpu = ggml_backend_cpu_init ();

  // Build backend pipeline list with CPU LAST
  std::vector<ggml_backend_t> backends;
  if (backend_gpu && ggml_backend_dev_type (ggml_backend_get_device (backend_gpu)) != GGML_BACKEND_DEVICE_TYPE_CPU)
    {
      backends.push_back (backend_gpu);
    }
  backends.push_back (backend_cpu);

  printf ("Using primary backend: %s\n", ggml_backend_name (backends[0]));

  // Allocate weight tensor context
  size_t                  ctx_size    = ggml_tensor_overhead () * 128;
  struct ggml_init_params init_params = { ctx_size, nullptr, true };
  struct ggml_context *   ctx_w       = ggml_init (init_params);

  ModelWeights weights;
  int          hd = params.n_embed / params.n_head;

  // Declare model parameters as F32
  weights.token_embed = ggml_new_tensor_2d (ctx_w, GGML_TYPE_F32, params.n_embed, params.n_vocab);
  weights.attn_norm   = ggml_new_tensor_1d (ctx_w, GGML_TYPE_F32, params.n_embed);
  weights.wq          = ggml_new_tensor_2d (ctx_w, GGML_TYPE_F32, params.n_embed, params.n_head * hd);
  weights.wk          = ggml_new_tensor_2d (ctx_w, GGML_TYPE_F32, params.n_embed, params.h_head_kv * hd);
  weights.wv          = ggml_new_tensor_2d (ctx_w, GGML_TYPE_F32, params.n_embed, params.h_head_kv * hd);
  weights.wo          = ggml_new_tensor_2d (ctx_w, GGML_TYPE_F32, params.n_head * hd, params.n_embed);
  weights.ffn_norm    = ggml_new_tensor_1d (ctx_w, GGML_TYPE_F32, params.n_embed);
  weights.w_gate      = ggml_new_tensor_2d (ctx_w, GGML_TYPE_F32, params.n_embed, params.n_ff);
  weights.w_up        = ggml_new_tensor_2d (ctx_w, GGML_TYPE_F32, params.n_embed, params.n_ff);
  weights.w_down      = ggml_new_tensor_2d (ctx_w, GGML_TYPE_F32, params.n_ff, params.n_embed);
  weights.output_norm = ggml_new_tensor_1d (ctx_w, GGML_TYPE_F32, params.n_embed);
  weights.output_w    = ggml_new_tensor_2d (ctx_w, GGML_TYPE_F32, params.n_embed, params.n_vocab);

  // Allocate memory buffer on chosen primary backend
  ggml_backend_buffer_t buffer_w = ggml_backend_alloc_ctx_tensors (ctx_w, backends[0]);

  // Fill model weights with mock data
  initialize_mock_weights (weights.token_embed);
  initialize_mock_weights (weights.attn_norm);
  initialize_mock_weights (weights.wq);
  initialize_mock_weights (weights.wk);
  initialize_mock_weights (weights.wv);
  initialize_mock_weights (weights.wo);
  initialize_mock_weights (weights.ffn_norm);
  initialize_mock_weights (weights.w_gate);
  initialize_mock_weights (weights.w_up);
  initialize_mock_weights (weights.w_down);
  initialize_mock_weights (weights.output_norm);
  initialize_mock_weights (weights.output_w);

  // Allocate dedicated KV Cache context & tensors [head_dim, h_head_kv, max_seq]
  struct ggml_context *ctx_kv = ggml_init (init_params);
  KVCache              kv;
  kv.k = ggml_new_tensor_3d (ctx_kv, GGML_TYPE_F32, hd, params.h_head_kv, params.max_seq);
  kv.v = ggml_new_tensor_3d (ctx_kv, GGML_TYPE_F32, hd, params.h_head_kv, params.max_seq);

  ggml_backend_buffer_t buffer_kv = ggml_backend_alloc_ctx_tensors (ctx_kv, backends[0]);
  ggml_backend_buffer_clear (buffer_kv, 0);

  // Setup execution scheduler with properly ordered backends (CPU last)
  ggml_backend_sched_t sched = ggml_backend_sched_new (
      backends.data (),
      nullptr,
      static_cast<int> (backends.size ()),
      GGML_DEFAULT_GRAPH_SIZE,
      false,
      false
      );

  int32_t prompt_token = 1550;
  int32_t n_past_pos   = 0;

  // Graph construction context
  size_t                  graph_buffer_size = ggml_tensor_overhead () * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead ();
  std::vector<uint8_t>    graph_buffer (graph_buffer_size);
  struct ggml_init_params graph_init_p = { graph_buffer_size, graph_buffer.data (), true };
  struct ggml_context *   ctx_graph    = ggml_init (graph_init_p);

  // Allocate input token & position tensor nodes
  struct ggml_tensor *inp_tokens = ggml_new_tensor_1d (ctx_graph, GGML_TYPE_I32, 1);
  struct ggml_tensor *inp_pos    = ggml_new_tensor_1d (ctx_graph, GGML_TYPE_I32, 1);

  // Build forward pass graph
  struct ggml_tensor *logits_tensor = nullptr;
  struct ggml_cgraph *gf            = build_transformer_graph (
      ctx_graph, params, weights, kv, inp_tokens, inp_pos, n_past_pos, &logits_tensor
      );

  // Allocate compute memory for dynamic graph nodes
  ggml_backend_sched_alloc_graph (sched, gf);

  // Assign input token & position data AFTER scheduler graph allocation
  ggml_backend_tensor_set (inp_tokens, &prompt_token, 0, sizeof (int32_t));
  ggml_backend_tensor_set (inp_pos, &n_past_pos, 0, sizeof (int32_t));

  printf ("Executing node-by-node forward pass graph compute...\n");
  ggml_backend_sched_graph_compute (sched, gf);

  // Fetch output logits from last node in execution graph
  std::vector<float> logits (params.n_vocab);
  ggml_backend_tensor_get (logits_tensor, logits.data (), 0, ggml_nbytes (logits_tensor));

  // Extract Top-5 Logits
  std::vector<int32_t> indices (params.n_vocab);
  std::iota (indices.begin (), indices.end (), 0);

  std::partial_sort (indices.begin (), indices.begin () + 5, indices.end (), [&logits](int32_t a, int32_t b)
  {
    return logits[a] > logits[b];
  });

  printf ("\n=== Top-5 Logits ===\n");
  for (int i = 0; i < 5; i++)
    {
      int32_t index = indices[i];
      printf ("Rank %d: Token ID = %6d | Logit = %8.4f\n", i + 1, index, logits[index]);
    }

  // Cleanup Resources
  ggml_backend_sched_free (sched);
  ggml_backend_buffer_free (buffer_kv);
  ggml_backend_buffer_free (buffer_w);
  ggml_free (ctx_graph);
  ggml_free (ctx_kv);
  ggml_free (ctx_w);

  if (backend_gpu && backend_gpu != backend_cpu)
    {
      ggml_backend_free (backend_gpu);
    }
  ggml_backend_free (backend_cpu);

  return 0;
}