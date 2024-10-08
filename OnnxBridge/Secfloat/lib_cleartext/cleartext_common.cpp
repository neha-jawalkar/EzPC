void NCHW_to_NHWC(int32_t N, int32_t C, int32_t H, int32_t W, vector<vector<vector<vector<float>>>> &inArr, vector<vector<vector<vector<float>>>> &outArr)
{
    for (uint32_t n = 0; n < N; n++)
    {
        for (uint32_t c = 0; c < C; c++)
        {
            for (uint32_t h = 0; h < H; h++)
            {
                for (uint32_t w = 0; w < W; w++)
                {
                    outArr[n][h][w][c] = inArr[n][c][h][w];
                }
            }
        }
    }
}

void NHWC_to_NCHW(int32_t N, int32_t H, int32_t W, int32_t C, vector<vector<vector<vector<float>>>> &inArr, vector<vector<vector<vector<float>>>> &outArr)
{
    for (uint32_t n = 0; n < N; n++)
    {
        for (uint32_t c = 0; c < C; c++)
        {
            for (uint32_t h = 0; h < H; h++)
            {
                for (uint32_t w = 0; w < W; w++)
                {
                    outArr[n][c][h][w] = inArr[n][h][w][c];
                }
            }
        }
    }
}

void OIFF_to_FFIO(int32_t CO, int32_t CI, int32_t FH, int32_t FW, vector<vector<vector<vector<float>>>> &inArr, vector<vector<vector<vector<float>>>> &outArr)
{
    for (uint32_t o = 0; o < CO; o++)
    {
        for (uint32_t i = 0; i < CI; i++)
        {
            for (uint32_t fh = 0; fh < FH; fh++)
            {
                for (uint32_t fw = 0; fw < FW; fw++)
                {
                    outArr[fh][fw][i][o] = inArr[o][i][fh][fw];
                }
            }
        }
    }
}

void __onnxbridge_MaxPool(int32_t N, int32_t C, int32_t H, int32_t W,
                          int32_t ksizeH, int32_t ksizeW,
                          int32_t strideH, int32_t strideW,
                          int32_t imgH, int32_t imgW,
                          vector<vector<vector<vector<float>>>> &inArr,
                          vector<vector<vector<vector<float>>>> &outArr,
                          int32_t padHLeft, int32_t padHRight, int32_t padWLeft, int32_t padWRight)
{
    vector<vector<vector<vector<float>>>> inputArr_reshaped = make_vector<float>(N, imgH, imgW, C);
    NCHW_to_NHWC(N, C, imgH, imgW, inArr, inputArr_reshaped);

    vector<vector<vector<vector<float>>>> outputArr_reshaped = make_vector<float>(N, H, W, C);
    MaxPool_nomask(N, imgH, imgW, C, ksizeH, ksizeW, strideH, strideW, H, W, inputArr_reshaped, outputArr_reshaped, padHLeft, padHRight, padWLeft, padWRight);

    NHWC_to_NCHW(N, H, W, C, outputArr_reshaped, outArr);
}

void __onnxbridge_AvgPool(int32_t N, int32_t C, int32_t H, int32_t W, int32_t ksizeH, int32_t ksizeW, int32_t strideH, int32_t strideW, int32_t imgH, int32_t imgW, vector<vector<vector<vector<float>>>> &inArr, vector<vector<vector<vector<float>>>> &outArr)
{
    vector<vector<vector<vector<float>>>> inputArr_reshaped = make_vector<float>(N, imgH, imgW, C) ;
    NCHW_to_NHWC(N, C, imgH, imgW, inArr, inputArr_reshaped) ;

    vector<vector<vector<vector<float>>>> outputArr_reshaped = make_vector<float>(N, H, W, C) ;
    AvgPool(N, imgH, imgW, C, ksizeH, ksizeW, strideH, strideW, H, W, inputArr_reshaped, outputArr_reshaped) ;

    NHWC_to_NCHW(N, H, W, C, outputArr_reshaped, outArr) ;
}

void __onnxbridge_Conv2DGroupWrapper(
    int32_t N, int32_t CI, int32_t H, int32_t W, int32_t FH, int32_t FW, int32_t CO,
    int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight,
    int32_t strideH, int32_t strideW, int32_t G,
    vector<vector<vector<vector<float>>>> &inputArr,
    vector<vector<vector<vector<float>>>> &filterArr,
    vector<vector<vector<vector<float>>>> &outArr)
{

    int32_t CIG = (CI / G);
    int32_t reshapedFilterRows = (CO / G);
    int32_t reshapedFilterCols = ((FH * FW) * CIG);
    int32_t reshapedIPRows = ((FH * FW) * CIG);
    int32_t outH = ((((H + (zPadHLeft + zPadHRight)) - FH) / strideH) + 1);
    int32_t outW = ((((W + (zPadWLeft + zPadWRight)) - FW) / strideW) + 1);
    int32_t reshapedIPCols = ((N * outH) * outW);

    vector<vector<vector<vector<float>>>> inputArr_reshaped = make_vector<float>(N, H, W, CI) ;
    NCHW_to_NHWC(N, CI, H, W, inputArr, inputArr_reshaped) ;

    vector<vector<vector<vector<float>>>> filterArr_reshaped = make_vector<float>(FH, FW, CI, CO);
    OIFF_to_FFIO(CO, CI, FH, FW, filterArr, filterArr_reshaped);

    vector<vector<vector<vector<float>>>> outputArr_reshaped = make_vector<float>(N, outH, outW, CO) ;
    Conv2DGroupWrapper(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, G, inputArr_reshaped, filterArr_reshaped, outputArr_reshaped);

    NHWC_to_NCHW(N, outH, outW, CO, outputArr_reshaped, outArr) ;
}

void __onnxbridge_ConvAdd(int32_t N, int32_t C, int32_t H, int32_t W,
             vector<vector<vector<vector<float>>>> &inArr,
             vector<float> &biasArr,
             vector<vector<vector<vector<float>>>> &outArr)
{
    vector<vector<vector<vector<float>>>> inputArr_reshaped = make_vector<float>(N, H, W, C) ;
    NCHW_to_NHWC(N, C, H, W, inArr, inputArr_reshaped) ;

    vector<vector<vector<vector<float>>>> outputArr_reshaped = make_vector<float>(N, H, W, C) ;
    ConvAdd(N, H, W, C, inputArr_reshaped, biasArr, outputArr_reshaped) ;

    NHWC_to_NCHW(N, H, W, C, outputArr_reshaped, outArr) ;
}