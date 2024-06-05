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
#include <sytorch/backend/piranha_cleartext.h>
#include <sytorch/backend/secureml_cleartext.h>
#include <sytorch/backend/float.h>
#include <sytorch/layers/layers.h>
#include <sytorch/module.h>
#include <llama/utils.h>
#include <llama/api.h>

bool hasPrinted = true;
bool hasInit = false;

template <typename T>
void printfe(Tensor<T> &t)
{
    if (hasInit) {
        std::cout << t.data[0] << std::endl;
    }
}

template <typename T>
class FFN : public SytorchModule<T>
{
    public:
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
    public:
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

public:
    FC<T> *c_attn;
    FC<T> *c_proj;

    u64 n_heads;
    u64 n_embd;

    MultiHeadAttention(u64 n_heads, u64 n_embd): n_heads(n_heads), n_embd(n_embd)
    {
        always_assert(n_embd % n_heads == 0);
        c_attn = new FC<T>(n_embd, 3*n_embd, true);
        c_proj = new FC<T>(n_embd, n_embd, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &x = c_attn->forward(input);
        auto &qkv_heads = split(x, 3);
        auto &q_heads = view(qkv_heads, 0);
        auto &k_heads = view(qkv_heads, 1);
        auto &v_heads = view(qkv_heads, 2);
        auto &qs = split(q_heads, n_heads);
        auto &ks = split(k_heads, n_heads);
        auto &vs = split(v_heads, n_heads);

        double divisor = 1 / sqrt(double(n_embd) / double(n_heads));

        std::vector<Tensor<T>*> qks_sm_vs;
        for(u64 i = 0; i < n_heads; ++i)
        {
            auto &q = view(qs, i);
            auto &k = view(ks, i);
            auto &v = view(vs, i);
            auto &kt = transpose(k);
            auto &qk = matmul(q, kt);
            auto &qks = scalarmul(qk, divisor);

            auto &qks_sm = softmax(qks);

            auto &qks_sm_v = matmul(qks_sm, v);
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
    public:
    using SytorchModule<T>::add;

    MultiHeadAttention<T> *attn;
    FFN<T> *ffn;
    LayerNorm<T> *ln0;
    LayerNorm<T> *ln1;
    
    u64 n_heads, n_embd;
public:

    TransformerBlock(u64 n_heads, u64 n_embd): n_heads(n_heads), n_embd(n_embd)
    {
        attn = new MultiHeadAttention<T>(n_heads, n_embd);
        ffn = new FFN<T>(n_embd, 4*n_embd);
        ln0 = new LayerNorm<T>(n_embd);
        ln1 = new LayerNorm<T>(n_embd);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &attn_out = attn->forward(input);
        auto &add0_out = add(attn_out, input);
        auto &ln0_out = ln0->forward(add0_out);

        auto &ffn_out = ffn->forward(ln0_out);
        auto &add1_out = add(ffn_out, ln0_out);
        auto &ln1_out = ln1->forward(add1_out);
        return ln1_out;
    }
};

template <typename T>
class BERT : public SytorchModule<T>
{
    public:
    using SytorchModule<T>::tanh;
    using SytorchModule<T>::view;
    using SytorchModule<T>::add;
    using SytorchModule<T>::unsqueeze;
    std::vector<TransformerBlock<T> *> blocks;
    LayerNorm<T> *ln_f;
    FC<T> *pool;
    u64 n_layer, n_heads, n_embd;

public:
    
    BERT(u64 n_layer, u64 n_heads, u64 n_embd): n_layer(n_layer), n_heads(n_heads), n_embd(n_embd)
    {
        for(u64 i = 0; i < n_layer; ++i)
        {
            blocks.push_back(new TransformerBlock<T>(n_heads, n_embd));
        }
        ln_f = new LayerNorm<T>(n_embd);
        pool = new FC<T>(n_embd, n_embd, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &y = ln_f->forward(input);
        Tensor<T> *x = &y;
        
        for(u64 i = 0; i < n_layer; ++i)
        {
            auto &block = blocks[i];
            auto &x_out = block->forward(*x);
            x = &x_out;
        }

        auto &x0 = view(*x, 0);
        auto &x0_unsqueeze = unsqueeze(x0);
        auto &pool_out = pool->forward(x0_unsqueeze);
        auto &tanh_out = tanh(pool_out);
        // return view(tanh_out, 0);
        return tanh_out;
    }
};


template <typename T>
class BERTSequenceClassification : public SytorchModule<T>
{
    public:
    using SytorchModule<T>::view;
    BERT<T> *gpt2;
    FC<T> *fc;
    u64 n_layer, n_heads, n_embd, n_labels;
public:
    
    BERTSequenceClassification(u64 n_layer, u64 n_heads, u64 n_embd, u64 n_labels): n_layer(n_layer), n_heads(n_heads), n_embd(n_embd), n_labels(n_labels)
    {
        gpt2 = new BERT<T>(n_layer, n_heads, n_embd);
        fc = new FC<T>(n_embd, n_labels, true);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &fc_in = gpt2->forward(input);
        auto &fc_out = fc->forward(fc_in);
        return view(fc_out, 0);
    }
};

u64 get_n_seq(std::string filename, u64 n_embd)
{
    u64 n_elements = std::filesystem::file_size(filename);
    assert(n_elements % (4 * n_embd) == 0);
    return n_elements / (4 * n_embd);
}

int validation(int __argc, char**__argv) {
    sytorch_init();

    std::string weights_and_dataset_pth = __argv[1];

    const u64 n_embd = 1024;
    const u64 n_head = n_embd / 64;
    const u64 n_layer = 24;
    const u64 scale = 12;
    const u64 bw = 51;

    using T = i64;

    BERTSequenceClassification<T> bert(n_layer, n_head, n_embd, 2);
    bert.init(scale);
    auto *ct = new ClearText<i64>();
    ct->bw = bw;
    bert.setBackend(ct);
    hasInit = true;
    bert.load(weights_and_dataset_pth + "/qnli/tiny/weights.dat");

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 5463; ++i) {
        std::string fname = std::string(weights_and_dataset_pth + "/qnli/tiny/dataset/") + std::to_string(i) + ".dat";
        u64 n_seq = get_n_seq(fname, n_embd);
        // std::cout << n_seq << std::endl;
        Tensor<T> input({n_seq, n_embd});
        input.load(fname, scale);
        auto &res = bert.forward(input);
        // print(bert.activation, scale);
        
        i64 max = INT_MIN;
        int argmax = 0;
        for(int i=0; i < 2; i++)
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
    // std::cout << "Total time = " << compute_time / (1000.0)  << " ms" << std::endl;

    return 0;
}

int main(int __argc, char**__argv)
{
    validation(__argc, __argv);
}
