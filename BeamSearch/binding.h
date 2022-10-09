﻿int paddle_beam_decode(THFloatTensor *th_probs,
                       THIntTensor *th_seq_lens,
                       size_t beam_size,
                       size_t num_processes,
                       size_t blank_id,
                       THIntTensor *th_output,
                       THIntTensor *th_timesteps,
                       THIntTensor *th_codeValues,
                       THFloatTensor *th_scores,
                       THIntTensor *th_out_length);