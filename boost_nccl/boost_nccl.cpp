#include <boost/python.hpp>
#include <vector>
#include <unordered_map>


// C interface to initialize the process groups, i.e. NCCL sub-communicators
extern "C" int horovod_nccl_create_process_groups(std::vector<std::vector<int32_t>> process_groups) ;
extern "C" std::vector<std::vector<int32_t>> horovod_process_groups() ;
/**
int horovod_nccl_create_process_groups(std::vector<std::vector<int32_t>> process_groups) {
	printf("horovod_nccl_create_process_groups() entered\n");
}
**/


// // C interface to reset the process groups, i.e. NCCL sub-communicators
int horovod_nccl_shutdown() {
  return 0;
}

namespace py = boost::python;

class boost_nccl{
    public:
	boost_nccl(){};
	void create_process_groups(boost::python::list& pylist);
	boost::python::list get_process_groups();
};

void boost_nccl::create_process_groups(boost::python::list& pylist) {
	printf("boost_nccl::create_process_groups() entered **************\n");
	int tmp;
	std::vector<std::vector<int>> vecvec;
	std::vector<int> vec;
	for (int i = 0; i < len(pylist) ; i++){
		vec.clear();
		for (int j = 0; j < len(pylist[i]) ; j++){
			tmp = boost::python::extract<int>(pylist[i][j]);
			vec.push_back(tmp);
		}
		vecvec.push_back(vec);
    	}
	for(int i=0; i<len(pylist) ; i++)
		for(int j=0; j<len(pylist[i]); j++)
			printf(" ******* vecvec[%d, %d] => %d *******\n", i, j, vecvec[i][j]);
	printf("boost_nccl::create_process_groups() calling horovod_nccl_create_process_groups()\n");
	horovod_nccl_create_process_groups(vecvec);
	printf("boost_nccl::create_process_groups() called horovod_nccl_create_process_groups()\n");
}

boost::python::list boost_nccl::get_process_groups()
{
	    boost::python::list pylist; // declare python list to return
	    printf("boost_nccl::get_process_groups() calling horovod_process_groups()***************\n");
	    std::vector<std::vector<int32_t>> pgs =  horovod_process_groups();
	    printf("boost_nccl::get_process_groups() called horovod_process_groups()\n");
	    for (int i=0; i<pgs.size(); i++) // loop over dim 0
	    {
		    boost::python::list tmp_list; // to temporary store 1d python list
	            for (int j=0; j<pgs[i].size(); j++) // loop over dim 1
	            {
		                tmp_list.append(pgs[i][j]); // append on temporary 1d python list
	            }
	            pylist.append(tmp_list); // append on 2d python list
	    }
	    return pylist; // return 2d python list
}


/// @brief Type that allows for registration of conversions from
///        python iterable types.
struct iterable_converter
{
  /// @note Registers converter from a python interable type to the
  ///       provided type.
  template <typename Container>
  iterable_converter&
  from_python()
  {
    boost::python::converter::registry::push_back(
      &iterable_converter::convertible,
      &iterable_converter::construct<Container>,
      boost::python::type_id<Container>());

    // Support chaining.
    return *this;
  }

  /// @brief Check if PyObject is iterable.
  static void* convertible(PyObject* object)
  {
    return PyObject_GetIter(object) ? object : NULL;
  }

  /// @brief Convert iterable PyObject to C++ container type.
  ///
  /// Container Concept requirements:
  ///
  ///   * Container::value_type is CopyConstructable.
  ///   * Container can be constructed and populated with two iterators.
  ///     I.e. Container(begin, end)
  template <typename Container>
  static void construct(
    PyObject* object,
    boost::python::converter::rvalue_from_python_stage1_data* data)
  {
    namespace python = boost::python;
    // Object is a borrowed reference, so create a handle indicting it is
    // borrowed for proper reference counting.
    python::handle<> handle(python::borrowed(object));

    // Obtain a handle to the memory block that the converter has allocated
    // for the C++ type.
    typedef python::converter::rvalue_from_python_storage<Container>
                                                                storage_type;
    void* storage = reinterpret_cast<storage_type*>(data)->storage.bytes;

    typedef python::stl_input_iterator<typename Container::value_type>
                                                                    iterator;

    // Allocate the C++ type into the converter's memory block, and assign
    // its handle to the converter's convertible variable.  The C++
    // container is populated by passing the begin and end iterators of
    // the python object to the container's constructor.
    new (storage) Container(
      iterator(python::object(handle)), // begin
      iterator());                      // end
    data->convertible = storage;
  }
};


BOOST_PYTHON_MODULE(boost_nccl){
        py::class_<boost_nccl>("boost_nccl", py::init<>())
          .def("create_process_groups", &boost_nccl::create_process_groups)
        ;

        //Register interable conversions.
        iterable_converter()
          .from_python<std::vector<int>>()
          .from_python<std::vector<std::vector<int>>>()
        ;

        py::def("create_process_groups", &boost_nccl::create_process_groups);
}
