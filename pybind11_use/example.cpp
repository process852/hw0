#include <pybind11/pybind11.h>

struct Pet{
    Pet(const std::string& name) : name (name){}
    void setName(const std::string& name_){
        this->name = name_;
    }
    const std::string& getName(){
        return this->name;
    }
    std::string name;
};

namespace py = pybind11;
PYBIND11_MODULE(example, m) {
    py::class_<Pet>(m, "Pet")
        .def(py::init<const std::string &>())
        .def("setName", &Pet::setName)
        .def("getName", &Pet::getName);
}