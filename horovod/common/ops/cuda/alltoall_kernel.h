namespace tensorflow {
	
namespace functor {
		
template <typename Device, typename T>
struct AlltoallFunctor {
	void operator()(const Device& d, int size, const T* in, T* out);
};
		
}  // namespace functor
	
}  // namespace tensorflowpe
