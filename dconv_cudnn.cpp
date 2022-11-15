#include <torch/extension.h>
#include <mutex>
#include <unordered_map>


#include <limits>
#include <vector>
#include <sstream>
#include <functional>
#include <ATen/cudnn/cudnn-wrapper.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>
// #include <c10/cuda/CUDACachingAllocator.h>

#include <unistd.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <cstdlib>
#include <atomic>

constexpr size_t operator"" _TiB(unsigned long long n)
{
  return size_t(n) * 1024 * 1024 * 1024 * 1024;
}
#define ASSERT_CORRECT_PRECISION(math_type)                                     \
if (args.params.dataType == CUDNN_DATA_FLOAT) {                                 \
  TORCH_INTERNAL_ASSERT(args.params.allow_tf32 || math_type == CUDNN_FMA_MATH); \
}
namespace at { namespace native {
std::string cudnnTypeToString(cudnnDataType_t dtype) {
  switch (dtype) {
    case CUDNN_DATA_FLOAT:
      return "CUDNN_DATA_FLOAT";
    case CUDNN_DATA_DOUBLE:
      return "CUDNN_DATA_DOUBLE";
    case CUDNN_DATA_HALF:
      return "CUDNN_DATA_HALF";
    case CUDNN_DATA_INT8:
      return "CUDNN_DATA_INT8";
    case CUDNN_DATA_INT32:
      return "CUDNN_DATA_INT32";
    case CUDNN_DATA_INT8x4:
      return "CUDNN_DATA_INT8x4";
    case CUDNN_DATA_UINT8:
      return "CUDNN_DATA_UINT8";
    case CUDNN_DATA_UINT8x4:
      return "CUDNN_DATA_UINT8x4";
    default:
      std::ostringstream oss;
      oss << "(unknown data-type " << static_cast<int>(dtype) << ")";
      return oss.str();
  }
}
std::string cudnnMemoryFormatToString(cudnnTensorFormat_t tformat) {
  switch (tformat) {
    case CUDNN_TENSOR_NCHW:
      return "CUDNN_TENSOR_NCHW";
    case CUDNN_TENSOR_NHWC:
      return "CUDNN_TENSOR_NHWC";
    default:
      std::ostringstream oss;
      oss << "(unknown cudnn tensor format " << static_cast<int>(tformat) << ")";
      return oss.str();
  }
}
std::ostream& operator<<(std::ostream & out, const TensorDescriptor& d) {
  out << "TensorDescriptor " << static_cast<void*>(d.desc()) << "\n";
  int nbDims;
  int dimA[CUDNN_DIM_MAX];
  int strideA[CUDNN_DIM_MAX];
  cudnnDataType_t dtype;
  cudnnGetTensorNdDescriptor(d.desc(), CUDNN_DIM_MAX, &dtype, &nbDims, dimA, strideA);
  out << "    type = " << cudnnTypeToString(dtype) << "\n";
  out << "    nbDims = " << nbDims << "\n";
  // Read out only nbDims of the arrays!
  out << "    dimA = ";
  for (auto i : ArrayRef<int>{dimA, static_cast<size_t>(nbDims)}) {
    out << i << ", ";
  }
  out << "\n";
  out << "    strideA = ";
  for (auto i : ArrayRef<int>{strideA, static_cast<size_t>(nbDims)}) {
    out << i << ", ";
  }
  out << "\n";
  return out;
}

void TensorDescriptor::print() { std::cout << *this; }

std::ostream& operator<<(std::ostream & out, const FilterDescriptor& d) {
  out << "FilterDescriptor " << static_cast<void*>(d.desc()) << "\n";
  int nbDims;
  int dimA[CUDNN_DIM_MAX];
  cudnnDataType_t dtype;
  cudnnTensorFormat_t tformat;
  cudnnGetFilterNdDescriptor(d.desc(), CUDNN_DIM_MAX, &dtype, &tformat, &nbDims, dimA);
  out << "    type = " << cudnnTypeToString(dtype) << "\n";
  out << "    tensor_format = " << cudnnMemoryFormatToString(tformat) << "\n";
  out << "    nbDims = " << nbDims << "\n";
  // Read out only nbDims of the arrays!
  out << "    dimA = ";
  for (auto i : ArrayRef<int>{dimA, static_cast<size_t>(nbDims)}) {
    out << i << ", ";
  }
  out << "\n";
  return out;
}

void FilterDescriptor::print() { std::cout << *this; }
constexpr int max_dim = 3;

constexpr int input_batch_size_dim = 0; // also grad_input
constexpr int input_channels_dim = 1;
constexpr int output_batch_size_dim = 0; // also grad_output
constexpr int output_channels_dim = 1;
constexpr int weight_output_channels_dim = 0;
constexpr int weight_input_channels_dim = 1;

template <typename Params>
struct ParamsHash {
  // Params must be a POD because we read out its memory
  // contenst as char* when hashing
  static_assert(std::is_pod<Params>::value, "Params is not POD");

  size_t operator()(const Params& params) const {
    auto ptr = reinterpret_cast<const uint8_t*>(&params);
    uint32_t value = 0x811C9DC5;
    for (int i = 0; i < (int)sizeof(Params); ++i) {
      value ^= ptr[i];
      value *= 0x01000193;
    }
    return (size_t)value;
  }
};

template <typename Params>
struct ParamsEqual {
  // Params must be a POD because we read out its memory
  // contenst as char* when comparing
  static_assert(std::is_pod<Params>::value, "Params is not POD");

  bool operator()(const Params& a, const Params& b) const {
    auto ptr1 = reinterpret_cast<const uint8_t*>(&a);
    auto ptr2 = reinterpret_cast<const uint8_t*>(&b);
    return memcmp(ptr1, ptr2, sizeof(Params)) == 0;
  }
};

static const std::array<cudnnConvolutionFwdAlgo_t, 8> fwd_algosd = {
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
};
static const std::array<cudnnConvolutionBwdDataAlgo_t, 6> bwd_algos = {
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED};
static const std::array<cudnnConvolutionBwdFilterAlgo_t, 6> bwd_w_algos = {
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING,
};

static inline at::MemoryFormat cudnn_conv_suggest_memory_format(const at::Tensor& input, const at::Tensor& weight) {
  // disable NHWC for float64 input.
  if (!at::detail::getCUDAHooks().compiledWithCuDNN() ||
      input.scalar_type() == at::kDouble ||
      weight.scalar_type() == at::kDouble) {
    return at::MemoryFormat::Contiguous;
  }
  long cudnn_version = at::detail::getCUDAHooks().versionCuDNN();
  auto input_memory_format = input.suggest_memory_format();
  auto weight_memory_format = weight.suggest_memory_format();
  auto weight_ndim = weight.ndimension();

  bool can_use_cudnn_channels_last_2d = (cudnn_version >= 7603) && (weight_ndim == 4) && (
    (input_memory_format  == at::MemoryFormat::ChannelsLast) ||
    (weight_memory_format == at::MemoryFormat::ChannelsLast)
  );
  if (can_use_cudnn_channels_last_2d) {
    return at::MemoryFormat::ChannelsLast;
  }

  bool can_use_cudnn_channels_last_3d = (cudnn_version >= 8005) && (weight_ndim == 5) && (
    (input_memory_format  == at::MemoryFormat::ChannelsLast3d) ||
    (weight_memory_format == at::MemoryFormat::ChannelsLast3d)
  );
  if (can_use_cudnn_channels_last_3d) {
    return at::MemoryFormat::ChannelsLast3d;
  }

  return at::MemoryFormat::Contiguous;
}

/* ---------------- for getCudnnDataType START ----------------*/
cudnnDataType_t getCudnnDataTypeFromScalarType(const ScalarType dtype) {
  if (dtype == c10::kQInt8) {
    return CUDNN_DATA_INT8;
  } else if (dtype == kFloat) {
    return CUDNN_DATA_FLOAT;
  } else if (dtype == kDouble) {
    return CUDNN_DATA_DOUBLE;
  } else if (dtype == kHalf) {
    return CUDNN_DATA_HALF;
  }
  std::string msg("getCudnnDataTypeFromScalarType() not supported for ");
  msg += toString(dtype);
  throw std::runtime_error(msg);
}

cudnnDataType_t getCudnnDataType(const Tensor& tensor) {
  return getCudnnDataTypeFromScalarType(tensor.scalar_type());
}

/* ---------------- for getCudnnDataType END ----------------*/


static inline bool cudnn_conv_use_channels_last(const at::Tensor &input, const at::Tensor &weight)
{
  // disable NHWC for float64 input.
  if (input.scalar_type() == at::kDouble ||
      weight.scalar_type() == at::kDouble)
  {
    return false;
  }
  return (CUDNN_VERSION >= 7603) &&
         ((input.suggest_memory_format() == at::MemoryFormat::ChannelsLast) ||
          (weight.suggest_memory_format() == at::MemoryFormat::ChannelsLast));
}

static void check_args(CheckedFrom c, IntArrayRef args, size_t expected_size, const char *arg_name)
{
  TORCH_CHECK(args.size() <= expected_size,
              "Too many ", arg_name, " values (", args.size(), ") supplied, expecting ",
              expected_size, " (while checking arguments for ", c, ")");
  TORCH_CHECK(args.size() >= expected_size,
              "Not enough ", arg_name, " values (", args.size(), ") supplied, expecting ",
              expected_size, " (while checking arguments for ", c, ")");

  auto num_negative_values = std::count_if(args.begin(), args.end(), [](int x)
                                           { return x < 0; });
  if (num_negative_values > 0)
  {
    std::stringstream ss;
    ss << arg_name << " should be greater than zero but got (";
    std::copy(args.begin(), args.end() - 1, std::ostream_iterator<int>(ss, ", "));
    ss << args.back() << ")"
       << " (while checking arguments for " << c << ")";
    AT_ERROR(ss.str());
  }
}
static void convolution_shape_check(
    CheckedFrom c,
    const TensorGeometryArg &input, const TensorGeometryArg &weight, const TensorGeometryArg &output,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups)
{
  check_args(c, padding, input->dim() - 2, "padding");
  check_args(c, stride, padding.size(), "stride");
  check_args(c, dilation, padding.size(), "dilation");

  // Input
  checkDimRange(c, input, 3, 6 /* exclusive */);
  checkSize(c, input, input_channels_dim, weight->size(1) * groups);

  // Weight
  checkSameDim(c, input, weight);

  // TODO: check that output->size() matches output_sizes
  // TODO: check that weight matches output->sizes()
  checkSameDim(c, input, output);
}

static inline std::vector<int64_t> conv_output_size(
    IntArrayRef input_size, IntArrayRef weight_size,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation = IntArrayRef())
{
  // ASSERT(input_size.size() > 2)
  // ASSERT(input_size.size() == weight_size.size())
  bool has_dilation = dilation.size() > 0;
  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);
  output_size[0] = input_size[input_batch_size_dim];
  output_size[1] = weight_size[weight_output_channels_dim];
  for (size_t d = 2; d < dim; ++d)
  {
    auto dilation_ = has_dilation ? dilation[d - 2] : 1;
    auto kernel = dilation_ * (weight_size[d] - 1) + 1;
    output_size[d] = (input_size[d] + (2 * padding[d - 2]) - kernel) / stride[d - 2] + 1;
  }
  return output_size;
}
static inline std::vector<int64_t> conv_input_size(
    IntArrayRef output_size, IntArrayRef weight_size,
    IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups)
{
  // ASSERT(output_size.size() > 2)
  // ASSERT(output_size.size() == weight_size.size())
  auto dim = output_size.size();
  std::vector<int64_t> input_size(dim);
  input_size[0] = output_size[output_batch_size_dim];
  input_size[1] = weight_size[weight_input_channels_dim] * groups;
  for (size_t d = 2; d < dim; ++d)
  {
    int kernel = dilation[d - 2] * (weight_size[d] - 1) + 1;
    input_size[d] = (output_size[d] - 1) * stride[d - 2] - (2 * padding[d - 2]) +
                    kernel + output_padding[d - 2];
  }
  return input_size;
}


struct ConvolutionParams
{
  c10::DeviceIndex device_id;
  cudnnDataType_t dataType;
  int input_size[2 + max_dim];
  uint8_t input_dim;
  at::MemoryFormat memory_format;
  int weight_size[2 + max_dim];
  int padding[max_dim];
  int stride[max_dim];
  int dilation[max_dim];
  int64_t groups;
  bool deterministic;
  bool allow_tf32;
  // NB: transposed purposely omitted: transposed just swaps
  // forward and backward, so you can reuse the benchmark entry,
};

// NB: This can't be a constructor, because then ConvolutionParams
// would not be a POD anymore.
// TODO: Use TensorGeometry here instead of the entire Tensor, which we
// don't actually need.  (OTOH: We can always pass in
// grad_input/grad_output, so this is not very pressing)
void setConvolutionParams(
    ConvolutionParams* params,
    const at::Tensor& input, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, 
    bool deterministic, bool allow_tf32, 
    at::MemoryFormat memory_format) {

  cudnnDataType_t dataType = getCudnnDataType(input);
  memset(params, 0, sizeof(ConvolutionParams));
  params->device_id = at::cuda::current_device();
  params->dataType = dataType;
  // ASSERT(weight.dim() == input.dim())
  params->input_dim = input.dim();
  params->memory_format = memory_format;
  for (int i = 0; i != params->input_dim; ++i) {
    params->input_size[i] = (int) input.sizes()[i];
    params->weight_size[i] = (int) weight.sizes()[i];
  }
  // ASSERT(padding.size() == stride.size())
  // ASSERT(padding.size() == dilation.size())
  for (size_t i = 0; i != padding.size(); ++i) {
    params->padding[i] = padding[i];
    params->stride[i] = stride[i];
    params->dilation[i] = dilation[i];
  }
  // In principle, we shouldn't parametrize by groups for legacy
  // CuDNN, but it doesn't seem worth the effort to actually do this.
  params->groups = groups;
  params->deterministic = deterministic;
  params->allow_tf32 = allow_tf32;
}


// Convenience struct for passing around descriptors and data
// pointers
struct ConvolutionArgs {
  cudnnHandle_t handle;
  ConvolutionParams params;
  TensorDescriptor idesc, odesc;
  FilterDescriptor wdesc;
  const Tensor& input, output, weight;
  ConvolutionDescriptor cdesc;

  ConvolutionArgs(const Tensor& input, const Tensor& output, const Tensor& weight) : input(input), output(output), weight(weight) {
  }
};
std::string repro_from_args(const ConvolutionParams& params) {
  auto pybool = [](bool b) -> const char* { return b ? "True" : "False"; };
  std::string partial_dtype;
  switch (params.dataType) {
    case CUDNN_DATA_FLOAT: partial_dtype = "float"; break;
    case CUDNN_DATA_DOUBLE: partial_dtype = "double"; break;
    case CUDNN_DATA_HALF: partial_dtype = "half"; break;
    default: partial_dtype = "unsupported";
  }
  const std::string full_dtype = "torch." + partial_dtype;
  const int out_channels = params.weight_size[0];
  const int in_channels = params.weight_size[1] * params.groups;
  const size_t dim = params.input_dim;
  const std::string channels_last_xd = dim == 4 ? "channels_last" : "channels_last_3d";
  const std::string to_channels_last = params.memory_format == at::MemoryFormat::ChannelsLast \
    ? ".to(memory_format=torch." + channels_last_xd + ")" : "";

  std::ostringstream ss;
  ss << "You can try to repro this exception using the following code snippet. ";
  ss << "If that doesn't trigger the error, please include your original repro script when reporting this issue.\n\n";
  ss << "import torch\n";
  ss << "torch.backends.cuda.matmul.allow_tf32 = " << pybool(at::globalContext().allowTF32CuBLAS()) << "\n";
  ss << "torch.backends.cudnn.benchmark = " << pybool(at::globalContext().benchmarkCuDNN()) << "\n";
  ss << "torch.backends.cudnn.deterministic = " << pybool(params.deterministic) << "\n";
  ss << "torch.backends.cudnn.allow_tf32 = " << pybool(params.allow_tf32) << "\n";
  ss << "data = torch.randn(" << ArrayRef<int>(params.input_size, dim) << ", dtype=" << full_dtype << ", ";
  ss <<   "device='cuda', requires_grad=True)" << to_channels_last << "\n";
  ss << "net = torch.nn.Conv" << dim-2 << "d(" << in_channels << ", " << out_channels << ", ";
  ss <<   "kernel_size=" << ArrayRef<int>(&params.weight_size[2], dim - 2) << ", ";
  ss <<   "padding=" << ArrayRef<int>(params.padding, dim-2) << ", ";
  ss <<   "stride=" << ArrayRef<int>(params.stride, dim-2) << ", ";
  ss <<   "dilation=" << ArrayRef<int>(params.dilation, dim-2) << ", ";
  ss <<   "groups=" << params.groups << ")\n";
  ss << "net = net.cuda()." << partial_dtype << "()" << to_channels_last << "\n";
  ss << "out = net(data)\n";
  ss << "out.backward(torch.randn_like(out))\n";
  ss << "torch.cuda.synchronize()\n\n";

  return ss.str();
}
std::ostream& operator<<(std::ostream & out, const ConvolutionArgs& args) {
  out << repro_from_args(args.params)  // already has a trailing newline
    << "input: " << args.idesc         // already has a trailing newline
    << "output: " << args.odesc        // already has a trailing newline
    << "weight: " << args.wdesc        // already has a trailing newline
    << "Pointer addresses: " << "\n"
    << "    input: " << args.input.data_ptr() << "\n"
    << "    output: " << args.output.data_ptr() << "\n"
    << "    weight: " << args.weight.data_ptr() << "\n";

  return out;
}

template <typename T>
struct BenchmarkCache {
  std::mutex mutex;
  std::atomic_flag bmk_cache_loaded = ATOMIC_FLAG_INIT;
  std::unordered_map<ConvolutionParams, T, ParamsHash<ConvolutionParams>, ParamsEqual<ConvolutionParams>> map;
  std::string CACHE_STORED_PATH;
  bool persist_cache = false;
  size_t searched = 0, hits = 0;

  BenchmarkCache(){
  }

  BenchmarkCache(std::string info_str){
    char *pcache = std::getenv("PERSIST_CACHE");
    if(pcache){
      if(std::stoi(pcache)){
        persist_cache = true;
      }
    }
    char *bmk_cache_path = std::getenv("BMK_CACHE_PATH");
    if(bmk_cache_path){
      CACHE_STORED_PATH = bmk_cache_path;
      CACHE_STORED_PATH = CACHE_STORED_PATH + "." + info_str;
    }else{
      CACHE_STORED_PATH = "temp_kv." + info_str;
    }
  }

  bool find(const ConvolutionParams& params, T* results) {
    std::lock_guard<std::mutex> guard(mutex);
    auto it = map.find(params);
    searched++;
    if (it == map.end()) {
      // std::cout<<CACHE_STORED_PATH<<":"<<hits<<"/"<<searched<<std::endl;
      return false;
    }
    hits++;
    *results = it->second;
    return true;
  }

  void insert(const ConvolutionParams& params, const T& results) {
    std::lock_guard<std::mutex> guard(mutex);
    map[params] = results;
  }

  int load(){
    std::lock_guard<std::mutex> guard(mutex);
    std::ifstream i_cache(CACHE_STORED_PATH, std::ios::in);
    if(i_cache){
      ConvolutionParams t_key;
      T t_ep;
      while(i_cache.read((char*)&t_key, sizeof(t_key)) && i_cache.read((char*)&t_ep, sizeof(t_ep))){
        map[t_key] = t_ep;
      }
      i_cache.close();
    }
    return 0;
  }

  int save(){
    std::lock_guard<std::mutex> guard(mutex);
    std::ofstream o_cache(CACHE_STORED_PATH, std::ios::out);
    for(auto &[t_key, t_ep] : map){
        o_cache.write((char*)&t_key, sizeof(t_key));
        o_cache.write((char*)&t_ep, sizeof(t_ep));
    }
    o_cache.close();
    return 0;
  }

  ~BenchmarkCache(){
    if(persist_cache){
      save();
      std::cout<<CACHE_STORED_PATH<<":"<<hits<<"/"<<searched<<std::endl;
    }
  }

};

BenchmarkCache<cudnnConvolutionFwdAlgoPerf_t> fwd_algos("fwd");
BenchmarkCache<cudnnConvolutionBwdDataAlgoPerf_t> bwd_data_algos("bwd_i");
BenchmarkCache<cudnnConvolutionBwdFilterAlgoPerf_t> bwd_filter_algos("bwd_w");

struct Workspace {
  Workspace(size_t size) : size(size), data(NULL) {
    // Sometimes cuDNN returns a workspace size > 2^63, this could makes the allocation of
    // workspace fail with some 64bit indexing error instead of an OOM error. In such case,
    // we manually fail with OOM.
    TORCH_CHECK_WITH(CUDAOutOfMemoryError, size < 1_TiB, "Not enough memory for workspace!");
    data = c10::cuda::CUDACachingAllocator::raw_alloc(size);
  }
  Workspace(const Workspace&) = delete;
  Workspace(Workspace&&) = default;
  Workspace& operator=(Workspace&&) = default;
  ~Workspace() {
    if (data) {
      c10::cuda::CUDACachingAllocator::raw_delete(data);
    }
  }

  size_t size;
  void* data;
};

template<typename perf_t>
struct algorithm_search {
};

cudnnStatus_t getWorkspaceSize(
    const ConvolutionArgs& args,
    cudnnConvolutionFwdAlgo_t algo, size_t* sz)
{
    return cudnnGetConvolutionForwardWorkspaceSize(
        args.handle,
        args.idesc.desc(),
        args.wdesc.desc(),
        args.cdesc.desc(),
        args.odesc.desc(),
        algo,
        sz
    );
}
cudnnStatus_t getWorkspaceSize(
    const ConvolutionArgs& args,
    cudnnConvolutionBwdDataAlgo_t algo, size_t* sz)
{
    return cudnnGetConvolutionBackwardDataWorkspaceSize(
        args.handle,
        args.wdesc.desc(),
        args.odesc.desc(),
        args.cdesc.desc(),
        args.idesc.desc(),
        algo,
        sz);
}
cudnnStatus_t getWorkspaceSize(
    const ConvolutionArgs& args,
    cudnnConvolutionBwdFilterAlgo_t algo, size_t* sz)
{
    return cudnnGetConvolutionBackwardFilterWorkspaceSize(
        args.handle,
        args.idesc.desc(),
        args.odesc.desc(),
        args.cdesc.desc(),
        args.wdesc.desc(),
        algo,
        sz);
}

template<typename algo_t>
size_t getMaxWorkspaceSize(
    const ConvolutionArgs& args,
    const algo_t *algo, int n_algo)
{
  size_t max_ws_size = 0;
  size_t max_block_size = 0;
  size_t tmp_bytes = 0;  // Only used for filling pointer parameters that aren't used later

  int device;
  THCudaCheck(cudaGetDevice(&device));
  c10::cuda::CUDACachingAllocator::cacheInfo(device, &tmp_bytes, &max_block_size);

  for (int i = 0; i < n_algo; i++) {
    cudnnStatus_t err;
    size_t sz;
    err = getWorkspaceSize(args, algo[i], &sz);
    if (CUDNN_STATUS_SUCCESS != err || sz == 0 || sz < max_ws_size || sz > max_block_size)
      continue;
    max_ws_size = sz;
  }
  return max_ws_size;
}

template<typename perf_t>
std::vector<perf_t> getValidAlgorithms(perf_t *perfResults, const ConvolutionArgs& args, int n_algo) {

// See Note [blocklist fft algorithms for strided dgrad]
// #if CUDNN_VERSION < 7500
//   bool blocklist = std::is_same<decltype(perfResults[0].algo), cudnnConvolutionBwdDataAlgo_t>::value;
//   int stride_dim = args.input.dim() - 2;
//   blocklist &= std::any_of(std::begin(args.params.stride),
//                             std::begin(args.params.stride) + stride_dim,
//                             [=](int n){return n != 1;});
// #endif

  std::vector<perf_t> result;
  result.reserve(n_algo);
  for (int i = 0; i < n_algo; i++) {
    perf_t perf = perfResults[i];

    // TODO: Shouldn't all returned results be successful?
    // Double check documentation for cudnnFindConvolutionForwardAlgorithmEx
    if (perf.status == CUDNN_STATUS_SUCCESS) {
      if (!args.params.deterministic || perf.determinism == CUDNN_DETERMINISTIC) {

        // See Note [blocklist fft algorithms for strided dgrad]
// #if CUDNN_VERSION < 7500
//         bool skip = blocklist;
//         skip &= (static_cast<cudnnConvolutionBwdDataAlgo_t>(perfResults[i].algo) == CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING ||
//                   static_cast<cudnnConvolutionBwdDataAlgo_t>(perfResults[i].algo) == CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT);
//         if (skip) {
//           continue;
//         }
// #endif

        result.push_back(perf);
      }
    }
  }
  TORCH_CHECK(result.size() > 0, "no valid convolution algorithms available in CuDNN");
  return result;
}

template<>
struct algorithm_search<cudnnConvolutionFwdAlgoPerf_t> {
  using perf_t = cudnnConvolutionFwdAlgoPerf_t;
  using algo_t = cudnnConvolutionFwdAlgo_t;

  static constexpr auto DEFAULT_ALGO = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  static BenchmarkCache<perf_t>& cache() { return fwd_algos; }

  static std::vector<perf_t> findAlgorithms(const ConvolutionArgs& args, bool benchmark) {
    static const algo_t algos[] = {
         CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
         CUDNN_CONVOLUTION_FWD_ALGO_FFT,
         CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
         CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
         CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
         CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
         CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
         CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
    };
    static constexpr int num_algos = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    static_assert(sizeof(algos) / sizeof(algos[0]) == num_algos,
                  "Missing cuDNN convolution forward algorithms");
    int perf_count;
    std::unique_ptr<perf_t[]> perf_results(new perf_t[num_algos]);
    if (!benchmark) {
      AT_CUDNN_CHECK_WITH_SHAPES(cudnnGetConvolutionForwardAlgorithm_v7(
          args.handle,
          args.idesc.desc(),
          args.wdesc.desc(),
          args.cdesc.desc(),
          args.odesc.desc(),
          num_algos,
          &perf_count,
          perf_results.get()), args);
    } else {
      size_t max_ws_size = getMaxWorkspaceSize(args, algos, num_algos);
      Workspace ws(max_ws_size);
      AT_CUDNN_CHECK_WITH_SHAPES(cudnnFindConvolutionForwardAlgorithmEx(
          args.handle,
          args.idesc.desc(), args.input.data_ptr(),
          args.wdesc.desc(), args.weight.data_ptr(),
          args.cdesc.desc(),
          args.odesc.desc(), args.output.data_ptr(),
          num_algos,
          &perf_count,
          perf_results.get(),
          ws.data,
          ws.size), args);

      // Free the cached blocks in our caching allocator. They are
      // needed here because the above benchmarking uses a huge amount of memory,
      // e.g. a few GBs.
      c10::cuda::CUDACachingAllocator::emptyCache();
    }
    return getValidAlgorithms<perf_t>(perf_results.get(), args, perf_count);
  }

  static void getWorkspaceSize(
    const ConvolutionArgs& args,
    algo_t algo, size_t* workspaceSize)
  {
    AT_CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        args.handle,
        args.idesc.desc(),
        args.wdesc.desc(),
        args.cdesc.desc(),
        args.odesc.desc(),
        algo,
        workspaceSize), args);
  }
};

template<>
struct algorithm_search<cudnnConvolutionBwdDataAlgoPerf_t> {
  using perf_t = cudnnConvolutionBwdDataAlgoPerf_t;
  using algo_t = cudnnConvolutionBwdDataAlgo_t;

  static constexpr auto DEFAULT_ALGO = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  static BenchmarkCache<perf_t>& cache() { return bwd_data_algos; }

  static std::vector<perf_t> findAlgorithms(const ConvolutionArgs& args, bool benchmark) {
    static const algo_t algos[] = {
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED
    };
    static constexpr int num_algos = CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
    static_assert(sizeof(algos) / sizeof(algos[0]) == num_algos,
                  "Missing cuDNN convolution backward data algorithms.");
    int perf_count;
    std::unique_ptr<perf_t[]> perf_results(new perf_t[num_algos]);
    if (!benchmark) {
      AT_CUDNN_CHECK_WITH_SHAPES(cudnnGetConvolutionBackwardDataAlgorithm_v7(
          args.handle,
          args.wdesc.desc(),
          args.odesc.desc(),
          args.cdesc.desc(),
          args.idesc.desc(),
          num_algos,
          &perf_count,
          perf_results.get()), args);
    } else {
      size_t max_ws_size = getMaxWorkspaceSize(args, algos, num_algos);
      Workspace ws(max_ws_size);
      AT_CUDNN_CHECK_WITH_SHAPES(cudnnFindConvolutionBackwardDataAlgorithmEx(
          args.handle,
          args.wdesc.desc(), args.weight.data_ptr(),
          args.odesc.desc(), args.output.data_ptr(),
          args.cdesc.desc(),
          args.idesc.desc(), args.input.data_ptr(),
          num_algos,
          &perf_count,
          perf_results.get(),
          ws.data,
          ws.size), args);

      // Free the cached blocks in our caching allocator. They are
      // needed here because the above benchmarking uses a huge amount of memory,
      // e.g. a few GBs.
      c10::cuda::CUDACachingAllocator::emptyCache();
    }
    return getValidAlgorithms<perf_t>(perf_results.get(), args, perf_count);
  }

  static void getWorkspaceSize(
    const ConvolutionArgs& args,
    cudnnConvolutionBwdDataAlgo_t algo, size_t* workspaceSize)
  {
    AT_CUDNN_CHECK_WITH_SHAPES(cudnnGetConvolutionBackwardDataWorkspaceSize(
        args.handle,
        args.wdesc.desc(),
        args.odesc.desc(),
        args.cdesc.desc(),
        args.idesc.desc(),
        algo,
        workspaceSize), args);
  }
};

template<>
struct algorithm_search<cudnnConvolutionBwdFilterAlgoPerf_t> {
  using perf_t = cudnnConvolutionBwdFilterAlgoPerf_t;
  using algo_t = cudnnConvolutionBwdFilterAlgo_t;

  static constexpr auto DEFAULT_ALGO = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

  static BenchmarkCache<perf_t>& cache() { return bwd_filter_algos; }

  static std::vector<perf_t> findAlgorithms(const ConvolutionArgs& args, bool benchmark) {
    static const algo_t algos[] = {
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING,
    };
    // NOTE: - 1 because ALGO_WINOGRAD is not implemented
    static constexpr int num_algos = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT - 1;
    static_assert(sizeof(algos) / sizeof(algos[0]) == num_algos,
                  "Missing cuDNN convolution backward filter algorithms.");
    std::unique_ptr<perf_t[]> perf_results(new perf_t[num_algos]);
    int perf_count;
    if (!benchmark) {
      AT_CUDNN_CHECK_WITH_SHAPES(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
          args.handle,
          args.idesc.desc(),
          args.odesc.desc(),
          args.cdesc.desc(),
          args.wdesc.desc(),
          num_algos,
          &perf_count,
          perf_results.get()), args);
    } else {
      size_t max_ws_size = getMaxWorkspaceSize(args, algos, num_algos);
      Workspace ws(max_ws_size);
      AT_CUDNN_CHECK_WITH_SHAPES(cudnnFindConvolutionBackwardFilterAlgorithmEx(
          args.handle,
          args.idesc.desc(), args.input.data_ptr(),
          args.odesc.desc(), args.output.data_ptr(),
          args.cdesc.desc(),
          args.wdesc.desc(), args.weight.data_ptr(),
          num_algos,
          &perf_count,
          perf_results.get(),
          ws.data,
          ws.size), args);

      // Free the cached blocks in our caching allocator. They are
      // needed here because the above benchmarking uses a huge amount of memory,
      // e.g. a few GBs.
      c10::cuda::CUDACachingAllocator::emptyCache();
    }
    return getValidAlgorithms<perf_t>(perf_results.get(), args, perf_count);
  }

  static void getWorkspaceSize(const ConvolutionArgs& args, algo_t algo, size_t* workspaceSize)
  {
    AT_CUDNN_CHECK_WITH_SHAPES(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        args.handle,
        args.idesc.desc(),
        args.odesc.desc(),
        args.cdesc.desc(),
        args.wdesc.desc(),
        algo,
        workspaceSize), args);
  }
};

template<typename perf_t>
class AlgoIterator {
  using search = algorithm_search<perf_t>;
  const ConvolutionArgs &args;
  bool benchmark;

public:
  AlgoIterator(const ConvolutionArgs &args, bool benchmark): args(args), benchmark(benchmark) {}

  static std::vector<perf_t> onlyDefaultAlgorithm(const ConvolutionArgs &args) {
    std::vector<perf_t> perfResults(1);
    perfResults[0].algo = search::DEFAULT_ALGO;
    if (args.params.dataType == CUDNN_DATA_HALF) {
      perfResults[0].mathType = CUDNN_TENSOR_OP_MATH;
    } else {
      perfResults[0].mathType = CUDNN_DEFAULT_MATH;
      if (args.params.dataType == CUDNN_DATA_FLOAT && !args.params.allow_tf32) {
        perfResults[0].mathType = CUDNN_FMA_MATH;
      }
    }
    search::getWorkspaceSize(args, perfResults[0].algo, &(perfResults[0].memory));
    return perfResults;
  }

  void try_all(std::function<void (const perf_t &perf)> f) {
    bool only_use_default = args.params.deterministic && !benchmark;
    auto& cache = search::cache();
    if(cache.persist_cache && !cache.bmk_cache_loaded.test_and_set()){
      cache.load();
    }
    perf_t algoPerf;
    if (!only_use_default && cache.find(args.params, &algoPerf)) {
      try {
        f(algoPerf);
        return;
      } catch (c10::CUDAOutOfMemoryError &e) {
        cudaGetLastError(); // clear CUDA error
      }
    }

    auto perfResults = only_use_default ? onlyDefaultAlgorithm(args) : search::findAlgorithms(args, benchmark);
    for (auto &algoPerf : perfResults) {
      try {
        f(algoPerf);
        cache.insert(args.params, algoPerf);
        return;
      } catch (c10::CUDAOutOfMemoryError &e) {
        cudaGetLastError(); // clear CUDA error
      } catch (c10::CuDNNError &e) {
        cudaGetLastError(); // clear CUDA error
      }
    }
    TORCH_CHECK(false, "Unable to find a valid cuDNN algorithm to run convolution");
  }
};

inline Tensor allocate_workspace(size_t size, const Tensor &other) {
  // Sometimes cuDNN returns a workspace size > 2^63, this could makes the allocation of
  // workspace fail with some 64bit indexing error instead of an OOM error. In such case,
  // we manually fail with OOM.
  TORCH_CHECK_WITH(CUDAOutOfMemoryError, size < 1_TiB, "Not enough memory for workspace!");
  return at::empty({static_cast<int64_t>(size)}, other.options().dtype(kByte));
}

// ---------------------------------------------------------------------
//
// Splitting to 32bit
//
// ---------------------------------------------------------------------

template <typename func_t, typename algo_t>
static inline void split_batch_dim_to_32bit_out(
    const at::Tensor &output,
    const at::Tensor &input,
    const at::Tensor &weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32,
    algo_t algo,
    int64_t max_worksize, func_t func_32bit)
{
  constexpr int64_t int_max = std::numeric_limits<int>::max();
  const int64_t ni = input.numel();
  const int64_t no = output.numel();
  // Assume the shape of the tensor is (N, C, D1, D2, ...)
  // if N * C * D1 * D2 * ... <= int_max, then no need to split at all
  if (ni <= int_max && no <= int_max)
  {
    func_32bit(
        output, input, weight, 
        padding, stride, dilation, groups, 
        benchmark, deterministic, allow_tf32,
        algo);
    return;
  }
  // else, if C * D1 * D2 * ... <= int_max, then we just need to split across the N dimension
  //
  // Here we use a simple heuristics to determine the size of each split
  // We don't max out the 2^31 address space because this number is super
  // large and very likely to get an OOM.
  int64_t n = output.size(0);
  int64_t max_inner_size = std::max<int64_t>(ni, no) / n;
  int64_t split_size = std::max<int64_t>(max_worksize / max_inner_size, 1L);
  int64_t num_splits = (n + split_size - 1) / split_size;
  if (split_size * max_inner_size < int_max)
  {
    for (int64_t i = 0; i < num_splits; i++)
    {
      int64_t start = split_size * i;
      int64_t split_size_ = std::min<int64_t>(split_size, n - start);
      Tensor input_ = input.narrow(0, start, split_size_);
      Tensor output_ = output.narrow(0, start, split_size_);
      func_32bit(
        output_, input_, weight, 
        padding, stride, dilation, groups, 
        benchmark, deterministic, allow_tf32,
        algo);
    }
    return;
  }
  // If control flow reaches here, this means even splitting N is not enough, then things starts to become complicated:
  // For example, for conv2d, there following questions needs to be considered.
  // - Is the memory layout NCHW or NHWC ?
  // - If the conv is NCHW -> NC'H'W', then should we
  //   - split only NC?
  //   - split only N'C'?
  //   - split both?
  // - If the conv is NHWC, then we need to split across H, we need to be very careful about the boundary condition
  //   to make sure that the boundary is handled correctly.
  // - If we decide to make these splits, is the memory contiguous? Do we need to copy the memory?
  // Considering the complexity of this issue, it is better not to use cuDNN for this case
  TORCH_INTERNAL_ASSERT(false, "This case should not be dispatched to cuDNN.");
}

// ---------------------------------------------------------------------
//
// Convolution forward / Transposed convolution backward
//
// ---------------------------------------------------------------------

// The raw API directly invokes CuDNN and does not emulate support
// for group convolution on old versions of CuDNN.
//
// There are a few reasons this should never be directly exposed
// via ATen:
//
//    - It takes output as a parameter (this should be computed!)
//    - It doesn't do input checking
//    - It doesn't resize output (it is assumed to be correctly sized)
//
void raw_cudnn_convolution_forward_out_32bit(
    const Tensor &output, const Tensor &input, const Tensor &weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32,
    cudnnConvolutionFwdAlgo_t algo) {

  auto dataType = getCudnnDataType(input);

  ConvolutionArgs args{input, output, weight};
  args.handle = getCudnnHandle();
  at::MemoryFormat memory_format = cudnn_conv_suggest_memory_format(input, weight);
  setConvolutionParams(&args.params, input, weight, padding, stride, dilation, groups, deterministic, allow_tf32, memory_format);
  args.idesc.set(input);
  args.wdesc.set(weight, 0, input.suggest_memory_format()==at::MemoryFormat::ChannelsLast);
  args.odesc.set(output);
  args.cdesc.set(dataType, input.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups, args.params.allow_tf32);
  // TODO: when we do legacy group convolution support, we'll repeatedly
  // reinitialize the workspace for each convolution we do.  This is
  // wasteful; we'd rather reuse the workspace.  OTOH, legacy group
  // convolution support is already pretty slow, so this might not
  // matter.  (This applies to raw_cudnn_convolution_backward_input as well.)
  AlgoIterator<cudnnConvolutionFwdAlgoPerf_t>(args, benchmark).try_all(
    [&](const cudnnConvolutionFwdAlgoPerf_t &fwdAlgPerf){
      Tensor workspace = allocate_workspace(fwdAlgPerf.memory, input);
  // std::cout<<"reached here!"<<std::endl;
      // update convDesc mathType since cudnn 7.4+ now requires both algo + mathType to figure out
      // whether to use Tensor core kernels or not
      // See Note [behavior of cudnnFind and cudnnGet]
      ASSERT_CORRECT_PRECISION(fwdAlgPerf.mathType);
      AT_CUDNN_CHECK(cudnnSetConvolutionMathType(args.cdesc.mut_desc(), fwdAlgPerf.mathType), args);

      Constant one(dataType, 1);
      Constant zero(dataType, 0);

      AT_CUDNN_CHECK(cudnnConvolutionForward(
          args.handle,
          &one, args.idesc.desc(), input.data_ptr(),
          args.wdesc.desc(), weight.data_ptr(),
          args.cdesc.desc(), fwdAlgPerf.algo, workspace.data_ptr(), fwdAlgPerf.memory,
          &zero, args.odesc.desc(), output.data_ptr()),
        args, "Forward algorithm: ", static_cast<int>(fwdAlgPerf.algo), "\n");
      }
  );
}

void raw_cudnn_convolution_forward_out(
    const Tensor &output, const Tensor &input, const Tensor &weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32,
    cudnnConvolutionFwdAlgo_t algo)
{
  split_batch_dim_to_32bit_out(
    output, 
    input, 
    weight, 
    padding, stride, dilation, groups, 
    benchmark, deterministic, allow_tf32,
    algo, 
    1024 * 1024 * 256, raw_cudnn_convolution_forward_out_32bit);
}

Tensor cudnn_convolution_forward(
    CheckedFrom c,
    const TensorArg& input, const TensorArg& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32,
    cudnnConvolutionFwdAlgo_t algo)
{
  checkAllSameType(c, {input, weight});
  checkAllSameGPU(c, {input, weight});

  auto memory_format = cudnn_conv_suggest_memory_format(*input, *weight);
  Tensor output_t = at::empty(
      conv_output_size(input->sizes(), weight->sizes(),
                       padding, stride, dilation),
      input->options().memory_format(memory_format));

  if (output_t.numel() == 0) {
    return output_t;
  }

  // Avoid ambiguity of "output" when this is being used as backwards
  TensorArg output{ output_t, "result", 0 };
  convolution_shape_check(c, input, weight, output, padding, stride, dilation, groups);

  Tensor weight_contig = weight->contiguous(memory_format);
  Tensor input_contig = input->contiguous(memory_format);

  raw_cudnn_convolution_forward_out(
      *output, input_contig, weight_contig,
      padding, stride, dilation, groups, 
      benchmark, deterministic, allow_tf32,
      algo);

  return *output;
}

Tensor cudnn_convolution_d(
    const Tensor &input_t, const Tensor &weight_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, int alg_type)
{
  const cudnnConvolutionFwdAlgo_t algo = (0 <= alg_type && alg_type < fwd_algosd.size()) ? fwd_algosd[alg_type] : CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  TensorArg input{input_t, "input", 1},
      weight{weight_t, "weight", 2};
  auto& ctx = at::globalContext();
  bool deterministic = ctx.deterministicCuDNN() || ctx.deterministicAlgorithms();
  auto output_t = cudnn_convolution_forward(
      "cudnn_convolution", 
      input, weight, 
      padding, stride, dilation, groups, 
      ctx.benchmarkCuDNN(), deterministic, ctx.allowTF32CuDNN(),
      algo);
  return output_t;
}

// ---------------------------------------------------------------------
//
// Convolution backward / Transposed convolution forward
//
// ---------------------------------------------------------------------

// ---------------------------------------------------------------------
//
// 1 - Convolution backward input(data)
//
// ---------------------------------------------------------------------

void raw_cudnn_convolution_backward_input_out_32bit(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32,
    cudnnConvolutionBwdDataAlgo_t algo) {
  auto dataType = getCudnnDataType(grad_output);

  ConvolutionArgs args{grad_input, grad_output, weight};
  args.handle = getCudnnHandle();
  at::MemoryFormat memory_format = cudnn_conv_suggest_memory_format(grad_input, weight);
  setConvolutionParams(&args.params, grad_input, weight, padding, stride, dilation, groups, deterministic, allow_tf32, memory_format);
  args.idesc.set(grad_input);
  args.wdesc.set(weight, 0, grad_output.suggest_memory_format()==at::MemoryFormat::ChannelsLast);
  args.odesc.set(grad_output);
  args.cdesc.set(dataType, grad_output.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups, args.params.allow_tf32);
  AlgoIterator<cudnnConvolutionBwdDataAlgoPerf_t>(args, benchmark).try_all(
    [&](const cudnnConvolutionBwdDataAlgoPerf_t &bwdDataAlgPerf){
      Tensor workspace = allocate_workspace(bwdDataAlgPerf.memory, grad_output);

      // update convDesc mathType since cudnn 7.4+ now requires both algo + mathType to figure out
      // whether to use Tensor core kernels or not
      // See Note [behavior of cudnnFind and cudnnGet]
      ASSERT_CORRECT_PRECISION(bwdDataAlgPerf.mathType);
      AT_CUDNN_CHECK(cudnnSetConvolutionMathType(args.cdesc.mut_desc(), bwdDataAlgPerf.mathType), args);

      Constant one(dataType, 1);
      Constant zero(dataType, 0);

      AT_CUDNN_CHECK(cudnnConvolutionBackwardData(
          args.handle,
          &one, args.wdesc.desc(), weight.data_ptr(),
          args.odesc.desc(), grad_output.data_ptr(),
          args.cdesc.desc(), bwdDataAlgPerf.algo, workspace.data_ptr(), bwdDataAlgPerf.memory,
          &zero, args.idesc.desc(), grad_input.data_ptr()),
        args,
        "Additional pointer addresses: \n",
        "    grad_output: ", grad_output.data_ptr(), "\n",
        "    grad_input: ", grad_input.data_ptr(), "\n",
        "Backward data algorithm: ", static_cast<int>(bwdDataAlgPerf.algo), "\n");
    }
  );
}

void raw_cudnn_convolution_backward_input_out(
    const at::Tensor &grad_input,
    const at::Tensor &grad_output,
    const at::Tensor &weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32,
    cudnnConvolutionBwdDataAlgo_t algo)
{
  split_batch_dim_to_32bit_out(
    grad_input, 
    grad_output, 
    weight, 
    padding, stride, dilation, groups,
    benchmark, deterministic, allow_tf32,
    algo, 
    1024 * 1024 * 128, raw_cudnn_convolution_backward_input_out_32bit);
}


Tensor cudnn_convolution_backward_input(
    CheckedFrom c,
    IntArrayRef input_size, const TensorArg &grad_output, const TensorArg &weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32,
    cudnnConvolutionBwdDataAlgo_t algo)
{
  checkAllSameType(c, {grad_output, weight});
  checkAllSameGPU(c, {grad_output, weight});

  // auto layout = cudnn_conv_use_channels_last(*grad_output, *weight) ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;
  auto layout = cudnn_conv_suggest_memory_format(*grad_output, *weight);
  auto grad_input_t = at::empty(input_size, grad_output->options(), layout);

  // Avoid "grad_input" when this is being used as transposed convolution
  TensorArg grad_input{grad_input_t, "result", 0};
  convolution_shape_check(c, grad_input, weight, grad_output, padding, stride, dilation, groups);

  // See #4500
  Tensor weight_contig = weight->contiguous(layout);
  // Make sure that NC11 strides follow formula
  // weight_contig.resize_(weight_contig.sizes(), layout);

  Tensor grad_output_contig = grad_output->contiguous(layout);
  // grad_output_contig.resize_(grad_output_contig.sizes(), layout);

  raw_cudnn_convolution_backward_input_out(
      *grad_input, grad_output_contig, weight_contig,
      padding, stride, dilation, groups,
      benchmark, deterministic, allow_tf32, 
      algo);

  return *grad_input;
}
Tensor cudnn_convolution_backward_input_d(
    IntArrayRef input_size, const Tensor &grad_output_t, const Tensor &weight_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    int alg_type)
{
  const cudnnConvolutionBwdDataAlgo_t algo = (0 <= alg_type && alg_type < bwd_algos.size()) ? bwd_algos[alg_type] : CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  TensorArg grad_output{grad_output_t, "grad_output", 1}, weight{weight_t, "weight", 2};
  auto& ctx = at::globalContext();
  bool deterministic = ctx.deterministicCuDNN() || ctx.deterministicAlgorithms();
  return cudnn_convolution_backward_input(
      "cudnn_convolution_backward_input",
      input_size, grad_output, weight,
      padding, stride, dilation, groups, 
      ctx.benchmarkCuDNN(), deterministic, ctx.allowTF32CuDNN(),
      algo);
}

// ---------------------------------------------------------------------
//
// Convolution backward (weight)
//
// ---------------------------------------------------------------------

void raw_cudnn_convolution_backward_weight_out_32bit(
    const Tensor& grad_weight, const Tensor& grad_output, const Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32,
    cudnnConvolutionBwdFilterAlgo_t algo) {

  auto dataType = getCudnnDataType(input);

  ConvolutionArgs args{input, grad_output, grad_weight};
  args.handle = getCudnnHandle();
  at::MemoryFormat memory_format = cudnn_conv_suggest_memory_format(input, grad_weight);
  setConvolutionParams(&args.params, input, grad_weight, padding, stride, dilation, groups, deterministic, allow_tf32, memory_format);
  args.idesc.set(input);
  args.wdesc.set(grad_weight, 0, input.suggest_memory_format()==at::MemoryFormat::ChannelsLast);
  args.odesc.set(grad_output);
  args.cdesc.set(dataType, input.dim() - 2, args.params.padding, args.params.stride, args.params.dilation, args.params.groups, args.params.allow_tf32);
  AlgoIterator<cudnnConvolutionBwdFilterAlgoPerf_t>(args, benchmark).try_all(
    [&](const cudnnConvolutionBwdFilterAlgoPerf_t &bwdFilterAlgPerf){
      Tensor workspace = allocate_workspace(bwdFilterAlgPerf.memory, input);

      // update convDesc mathType since cudnn 7.4+ now requires both algo + mathType to figure out
      // whether to use Tensor core kernels or not
      // See Note [behavior of cudnnFind and cudnnGet]
      ASSERT_CORRECT_PRECISION(bwdFilterAlgPerf.mathType);
      AT_CUDNN_CHECK(cudnnSetConvolutionMathType(args.cdesc.mut_desc(), bwdFilterAlgPerf.mathType), args);

      Constant one(dataType, 1);
      Constant zero(dataType, 0);

      AT_CUDNN_CHECK(cudnnConvolutionBackwardFilter(
          args.handle,
          &one, args.idesc.desc(), input.data_ptr(),
          args.odesc.desc(), grad_output.data_ptr(),
          args.cdesc.desc(), bwdFilterAlgPerf.algo, workspace.data_ptr(), bwdFilterAlgPerf.memory,
          &zero, args.wdesc.desc(), grad_weight.data_ptr()),
        args,
        "Additional pointer addresses: \n",
        "    grad_output: ", grad_output.data_ptr(), "\n",
        "    grad_weight: ", grad_weight.data_ptr(), "\n",
        "Backward filter algorithm: ", static_cast<int>(bwdFilterAlgPerf.algo), "\n");
    }
  );
}

void raw_cudnn_convolution_backward_weight_out(
    const Tensor &grad_weight, const Tensor &grad_output, const Tensor &input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32,
    cudnnConvolutionBwdFilterAlgo_t algo)
{
  constexpr int64_t int_max = std::numeric_limits<int>::max();
  const int64_t ni = input.numel();
  const int64_t no = grad_output.numel();
  // Assume the shape of the tensor is (N, C, D1, D2, ...)
  // if N * C * D1 * D2 * ... <= int_max, then no need to split at all
  if (ni <= int_max && no <= int_max)
  {
    raw_cudnn_convolution_backward_weight_out_32bit(
      grad_weight, grad_output, input, 
      padding, stride, dilation, groups, 
      benchmark, deterministic, allow_tf32,
      algo);
    return;
  }
  // else, if C * D1 * D2 * ... <= int_max, then we just need to split across the N dimension
  //
  // Here we use a simple heuristics to determine the size of each split
  // We don't max out the 2^31 address space because this number is super
  // large and very likely to get an OOM.
  int64_t n = grad_output.size(0);
  int64_t max_inner_size = std::max<int64_t>(ni, no) / n;
  int64_t split_size = std::max<int64_t>(1024 * 1024 * 512 / max_inner_size, 1L);
  int64_t num_splits = (n + split_size - 1) / split_size;
  if (split_size * max_inner_size < int_max)
  {
    for (int64_t i = 0; i < num_splits; i++)
    {
      int64_t start = split_size * i;
      int64_t split_size_ = std::min<int64_t>(split_size, n - start);
      Tensor input_ = input.narrow(0, start, split_size_);
      Tensor grad_output_ = grad_output.narrow(0, start, split_size_);
      Tensor grad_weight_ = at::empty_like(grad_weight);
      raw_cudnn_convolution_backward_weight_out_32bit(
        grad_weight_, grad_output_, input_, 
        padding, stride, dilation, groups, 
        benchmark, deterministic, allow_tf32,
        algo);
      grad_weight.add_(grad_weight_);
    }
    return;
  }
  // If control flow reaches here, this means even splitting N is not enough, then things starts to become complicated:
  // For example, for conv2d, there following questions needs to be considered.
  // - Is the memory layout NCHW or NHWC ?
  // - If the conv is NCHW -> NC'H'W', then should we
  //   - split only NC?
  //   - split only N'C'?
  //   - split both?
  // - If the conv is NHWC, then we need to split across H, we need to be very careful about the boundary condition
  //   to make sure that the boundary is handled correctly.
  // - If we decide to make these splits, is the memory contiguous? Do we need to copy the memory?
  // Considering the complexity of this issue, it is better not to use cuDNN for this case
  TORCH_INTERNAL_ASSERT(false, "This case should not be dispatched to cuDNN.");
}

Tensor cudnn_convolution_backward_weight(
    CheckedFrom c,
    IntArrayRef weight_size, const Tensor &grad_output_t, const Tensor &input_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32,
    cudnnConvolutionBwdFilterAlgo_t algo)
{
  auto layout = cudnn_conv_suggest_memory_format(input_t, grad_output_t);

  Tensor grad_output_contig_t = grad_output_t.contiguous(layout);
  // Make sure that NC11 strides follow formula
  // grad_output_contig_t.resize_(grad_output_contig_t.sizes(), layout);
  TensorArg grad_output_contig{grad_output_contig_t, "grad_output", 1};

  Tensor input_contig_t = input_t.contiguous(layout);
  // input_contig_t.resize_(input_contig_t.sizes(), layout);
  TensorArg input{input_contig_t, "input", 2};

  checkAllSameType(c, {grad_output_contig, input});
  checkAllSameGPU(c, {grad_output_contig, input});

  auto grad_weight_t = at::empty(weight_size, grad_output_contig->options(), layout);

  // For uniformity with everything else, although it seems grad_weight
  // would be unambiguous too.
  TensorArg grad_weight{grad_weight_t, "result", 0};
  convolution_shape_check(c, input, grad_weight, grad_output_contig, padding, stride, dilation, groups);

  raw_cudnn_convolution_backward_weight_out(
      *grad_weight, *grad_output_contig, *input,
      padding, stride, dilation, groups, 
      benchmark, deterministic, allow_tf32,
      algo);

  return grad_weight_t;
}

Tensor cudnn_convolution_backward_weight_d(
    IntArrayRef weight_size,
    const Tensor &grad_output_t,
    const Tensor &input_t,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    int alg_type)
{
  const cudnnConvolutionBwdFilterAlgo_t algo = (0 <= alg_type && alg_type < bwd_w_algos.size()) ? bwd_w_algos[alg_type] : CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
  auto& ctx = at::globalContext();
  bool deterministic = ctx.deterministicCuDNN() || ctx.deterministicAlgorithms();
  return cudnn_convolution_backward_weight(
      "cudnn_convolution_backward_weight",
      weight_size, grad_output_t, input_t,
      padding, stride, dilation, groups, 
      ctx.benchmarkCuDNN(), deterministic, ctx.allowTF32CuDNN(),
      algo);
}
}}  // namespace at::native



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("n_fwd_algos", []()
        { return at::native::fwd_algosd.size(); });
  m.def("n_bwd_ip_algos", []()
        { return at::native::bwd_algos.size(); });
  m.def("n_bwd_wt_algos", []()
        { return at::native::bwd_w_algos.size(); });
  m.def("cudnn_convolution", &at::native::cudnn_convolution_d);
  m.def("cudnn_convolution_backward_input", &at::native::cudnn_convolution_backward_input_d);
  m.def("cudnn_convolution_backward_weight", &at::native::cudnn_convolution_backward_weight_d);

}
