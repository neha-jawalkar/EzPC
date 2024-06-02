// Authors: Kanav Gupta, Neha Jawalkar
// Copyright:
// 
// Copyright (c) 2024 Microsoft Research
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <sytorch/backend/llama_extended.h>
#include <sytorch/backend/llama_transformer.h>
#include <sytorch/backend/baseline_cleartext.h>
#include <sytorch/layers/layers.h>
#include <sytorch/module.h>
#include <llama/utils.h>
#include <llama/api.h>

bool hasInit = false;

template <typename T>
class FFN : public SytorchModule<T>
{
    using SytorchModule<T>::gelu;

    u64 in;
    u64 hidden;
public:
    FC<T> *up;
    FC<T> *down;

    FFN(u64 in, u64 hidden) : in(in), hidden(hidden) 
    {
        up = new FC<T>(in, hidden, true);
        down = new FC<T>(hidden, in, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        return down->forward(gelu(up->forward(input)));
    }
};

template <typename T>
class MultiHeadAttention : public SytorchModule<T>
{
    using SytorchModule<T>::split;
    using SytorchModule<T>::view;
    using SytorchModule<T>::add;
    using SytorchModule<T>::transpose;
    using SytorchModule<T>::matmul;
    using SytorchModule<T>::scalarmul;
    using SytorchModule<T>::invsqrt;
    using SytorchModule<T>::softmax;
    using SytorchModule<T>::concat;
    using SytorchModule<T>::attention_mask;
    using SytorchModule<T>::local_attention_mask;
    ///////////////////////////
    using SytorchModule<T>::matmul_triangular;
    using SytorchModule<T>::softmax_triangular;

public:
    // FC<T> *c_attn;
    FC<T> *k_attn;
    FC<T> *v_attn;
    FC<T> *q_attn;
    FC<T> *c_proj;

    u64 n_heads;
    u64 n_embd;
    u64 attention_type;
    u64 window_size;

    MultiHeadAttention(u64 n_heads, u64 n_embd, u64 attention_type, u64 window_size): n_heads(n_heads), n_embd(n_embd)
    {
        always_assert(n_embd % n_heads == 0);
        // c_attn = new FC<T>(n_embd, 3*n_embd, true);
        k_attn = new FC<T>(n_embd, n_embd, false);
        v_attn = new FC<T>(n_embd, n_embd, false);
        q_attn = new FC<T>(n_embd, n_embd, false);
        c_proj = new FC<T>(n_embd, n_embd, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        // auto &x = c_attn->forward(input);
        // auto &qkv_heads = split(x, 3);
        // auto &q_heads = view(qkv_heads, 0);
        // auto &k_heads = view(qkv_heads, 1);
        // auto &v_heads = view(qkv_heads, 2);
        auto &k_heads = k_attn->forward(input);
        auto &v_heads = v_attn->forward(input);
        auto &q_heads = q_attn->forward(input);
        auto &qs = split(q_heads, n_heads);
        auto &ks = split(k_heads, n_heads);
        auto &vs = split(v_heads, n_heads);

        // double divisor = 1 / sqrt(double(n_embd) / double(n_heads));
        // double divisor = 1;

        std::vector<Tensor<T>*> qks_sm_vs;
        for(u64 i = 0; i < n_heads; ++i)
        {
            auto &q = view(qs, i);
            auto &k = view(ks, i);
            auto &v = view(vs, i);
            auto &kt = transpose(k);
            auto &qks = matmul(q, kt);

            // auto &qks = matmul_triangular(q, kt);

            // auto &qk = matmul(q, kt);
            // auto &qks = scalarmul(qk, divisor);

            
            Tensor<T> *x = &input;
            if(attention_type % 2 == 0)
            {   
                auto &qks_masked = attention_mask(qks, 10000000.0);
                x = &qks_masked;
            }
            else 
            {
                auto &qks_masked = local_attention_mask(qks, 10000000.0);
                x = &qks_masked;
            }
            auto &qks_sm = softmax(*x);
            auto &qks_sm_v = matmul(qks_sm, v);
            

        //    Tensor<T> *x = &input;
        //     if(attention_type % 2 == 0)
        //     {   
        //         auto &qks_sm = softmax_triangular(qks);
        //         x = &qks_sm;
        //     }
        //     else 
        //     {
        //         auto &qks_masked = local_attention_mask(qks, 10000.0);
        //         auto &qks_sm = softmax_triangular(qks_masked);
        //         x = &qks_sm;
        //     }
        //     auto &qks_sm_v = matmul(*x, v);

            qks_sm_vs.push_back(&qks_sm_v);
        }

        auto &qks_sm_vs_cat = concat(qks_sm_vs);
        auto &res = c_proj->forward(qks_sm_vs_cat);
        return res;
    }
};

template <typename T>
class TransformerBlock : public SytorchModule<T>
{
    using SytorchModule<T>::add;

    MultiHeadAttention<T> *attn;
    FFN<T> *ffn;
    LayerNorm<T> *ln0;
    LayerNorm<T> *ln1;
    
    u64 n_heads, n_embd;
    u64 attention_type; 
    u64 window_size;
public:

    TransformerBlock(u64 n_heads, u64 n_embd, u64 attention_type, u64 window_size): n_heads(n_heads), n_embd(n_embd)
    {
        attn = new MultiHeadAttention<T>(n_heads, n_embd, attention_type, window_size);
        ffn = new FFN<T>(n_embd, 4*n_embd);
        ln0 = new LayerNorm<T>(n_embd);
        ln1 = new LayerNorm<T>(n_embd);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &ln0_out = ln0->forward(input);
        auto &attn_out = attn->forward(ln0_out);
        auto &attn_out_add = add(attn_out, input);
        auto &ln1_out = ln1->forward(attn_out_add);
        auto &ffn_out = ffn->forward(ln1_out);
        auto &ffn_out_add = add(ffn_out, attn_out_add);
        return ffn_out_add;
    }
};

template <typename T>
class GPT2 : public SytorchModule<T>
{
    std::vector<TransformerBlock<T> *> blocks;
    LayerNorm<T> *ln_f;
    u64 n_layer, n_heads, n_embd;
    u64 window_size;

public:
    
    GPT2(u64 n_layer, u64 n_heads, u64 n_embd, u64 window_size): n_layer(n_layer), n_heads(n_heads), n_embd(n_embd)
    {
        for(u64 i = 0; i < n_layer; ++i)
        {
            blocks.push_back(new TransformerBlock<T>(n_heads, n_embd, i, window_size));
        }
        ln_f = new LayerNorm<T>(n_embd);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        Tensor<T> *x = &input;

        // for(u64 i = 0; i < n_layer - 1; ++i)
        // {
        //     auto &block = blocks[i];
        //     auto &x_out = block->forward(*x);
        //     x = &x_out;
        // }

        // auto &block = blocks[n_layer - 1];
        // return block->forward(*x);
        
        for(u64 i = 0; i < n_layer; ++i)
        {
            auto &block = blocks[i];
            auto &x_out = block->forward(*x);
            x = &x_out;
        }
        return ln_f->forward(*x);
    }
};

u64 get_n_seq(std::string filename, u64 n_embd)
{
    u64 n_elements = std::filesystem::file_size(filename);
    assert(n_elements % (4 * n_embd) == 0);
    return n_elements / (4 * n_embd);
}


template <typename T>
class GPT2NextWordLogits : public SytorchModule<T>
{
    using SytorchModule<T>::view;
    using SytorchModule<T>::transpose;
    GPT2<T> *gpt2;
    FC<T> *fc;
    u64 n_layer, n_heads, n_embd, n_vocab;
    u64 window_size;
public:
    
    GPT2NextWordLogits(u64 n_layer, u64 n_heads, u64 n_embd, u64 n_vocab, u64 window_size): n_layer(n_layer), n_heads(n_heads), n_embd(n_embd), n_vocab(n_vocab)
    {
        gpt2 = new GPT2<T>(n_layer, n_heads, n_embd, window_size);
        fc = new FC<T>(n_embd, n_vocab, false);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &fc_in = gpt2->forward(input);
        // printshape(fc_in.shape);
        auto &fc_out = fc->forward(fc_in);
        return view(fc_out, -1);
    }
};


int test_gpt2NextWordLogits_ct() {
    sytorch_init();

    const u64 n_vocab = 50257;
    const u64 n_ctx = 2048;
    const u64 n_embd = 2048;
    const u64 n_head = 16;
    const u64 n_layer = 24;
    const u64 scale = 12;
    const u64 window_size = 256;

    // GPT2<i64> gpt2(n_layer, n_head, n_embd, window_size);
    GPT2NextWordLogits<i64> gpt2(n_layer, n_head, n_embd, n_vocab, window_size);
    gpt2.init(scale);
    auto *ct = new ClearText<i64>();
    ct->bw = 51;
    gpt2.setBackend(ct);
    gpt2.load("/mnt/nvme/kanav/sigma-accuracy/lambada/gptneo/weights.dat");

    auto t1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 4; ++i) {
        std::string fname = std::string("/mnt/nvme/kanav/sigma-accuracy/lambada/gptneo/dataset/") + std::to_string(i) + ".dat";

        u64 n_seq = get_n_seq(fname, n_embd);
        Tensor<i64> input({n_seq, n_embd});
        input.load(fname, scale);

        auto &res = gpt2.forward(input);

        i64 max = INT_MIN;
        int argmax = 0;
        for(int i=0; i<50257; i++)
        {   
            if(res.data[i]>max)
            {
                max = res.data[i];
                argmax = i;
            }
        }
        std::cout << argmax << std::endl;
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Total time = " << compute_time / (1000.0)  << " ms" << std::endl;

    return 0;
}

int main(int __argc, char**__argv)
{
    test_gpt2NextWordLogits_ct();
}
