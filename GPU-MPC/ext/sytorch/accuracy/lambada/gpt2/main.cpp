#include <sytorch/backend/llama_extended.h>
#include <sytorch/backend/llama_transformer.h>
#include <sytorch/backend/piranha_cleartext.h>
#include <sytorch/backend/secureml_cleartext.h>
#include <sytorch/backend/crypten_cleartext.h>
#include <sytorch/layers/layers.h>
#include <sytorch/module.h>
#include <llama/utils.h>
#include <llama/api.h>

Tensor<i64> toi64(Tensor<u64> &t, u64 bw)
{
    Tensor<i64> res(t.shape);
    
    for (int i = 0; i < t.size(); ++i)
        res.data[i] = (t.data[i] + (1LL << (bw - 1))) % (1LL << bw) - (1LL << (bw - 1));
    return res;
}

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
    using SytorchModule<T>::transpose;
    using SytorchModule<T>::matmul;
    using SytorchModule<T>::matmul_triangular;
    using SytorchModule<T>::scalarmul;
    using SytorchModule<T>::softmax_triangular;
    using SytorchModule<T>::concat;

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
            auto &qk = matmul_triangular(q, kt);
            auto &qks = scalarmul(qk, divisor);

            auto &qks_sm = softmax_triangular(qks);

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

public:
    
    GPT2(u64 n_layer, u64 n_heads, u64 n_embd): n_layer(n_layer), n_heads(n_heads), n_embd(n_embd)
    {
        for(u64 i = 0; i < n_layer; ++i)
        {
            blocks.push_back(new TransformerBlock<T>(n_heads, n_embd));
        }
        ln_f = new LayerNorm<T>(n_embd);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        Tensor<T> *x = &input;
        
        for(u64 i = 0; i < n_layer; ++i)
        {
            auto &block = blocks[i];
            auto &x_out = block->forward(*x);
            x = &x_out;
        }
        return ln_f->forward(*x);
    }
};

template <typename T>
class GPT2SequenceClassification : public SytorchModule<T>
{
    using SytorchModule<T>::view;
    GPT2<T> *gpt2;
    FC<T> *fc;
    u64 n_layer, n_heads, n_embd, n_labels;
public:
    
    GPT2SequenceClassification(u64 n_layer, u64 n_heads, u64 n_embd, u64 n_labels): n_layer(n_layer), n_heads(n_heads), n_embd(n_embd), n_labels(n_labels)
    {
        gpt2 = new GPT2<T>(n_layer, n_heads, n_embd);
        fc = new FC<T>(n_embd, n_labels, false);
    }

    Tensor<T> &_forward(Tensor<T> &input)
    {
        auto &fc_in = gpt2->forward(input);
        auto &fc_out = fc->forward(fc_in);
        return view(fc_out, -1);
        // return fc_in;
    }
};

u64 get_n_seq(std::string filename, u64 n_embd)
{
    u64 n_elements = std::filesystem::file_size(filename);
    assert(n_elements % (4 * n_embd) == 0);
    return n_elements / (4 * n_embd);
}


void ct_main() {
    sytorch_init();

    const u64 n_vocab = 50257;
    const u64 n_ctx = 1024;
    const u64 n_embd = 768;
    const u64 n_head = 12;
    const u64 n_layer = 12;
    const u64 scale = 12;

    GPT2SequenceClassification<i64> gpt2(n_layer, n_head, n_embd, n_vocab);
    auto *ct = new ClearText<i64>();
    ct->bw = 51;
    gpt2.init(scale);
    gpt2.setBackend(ct);
    gpt2.load("/home/kanav/sigma-accuracy/lambada/gpt2/weights.dat");

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 5153; ++i) {
        std::string fname = std::string("/home/kanav/sigma-accuracy/lambada/gpt2/dataset/") + std::to_string(i) + ".dat";
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
}

int main(int __argc, char**__argv)
{
    ct_main();
    return 0;
}