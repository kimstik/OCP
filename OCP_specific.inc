#pragma once
#include <Standard_Handle.hxx>
#include <type_traits>
#include <memory>
#include <fmt/format.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

PYBIND11_DECLARE_HOLDER_TYPE(T, opencascade::handle<T>, true);

using opencascade::handle;

template <typename CppException>
void register_occ_exception(py::handle scope,
                        const char *name){

    py::register_exception<CppException>(scope, name);
};


template <typename T, typename Ptr>
inline void register_default_constructor(py::module m, const char* name){

    if constexpr (std::is_constructible<T>::value){
        static_cast<py::class_<T ,Ptr>>(m.attr(name))
            .def(py::init([]() { return new T; }));
    }

};

template <typename T>
inline void copy_if_assignable(T& t1, T& t2){

    if constexpr (std::is_assignable<T, T>::value){
        t1 = t2;
    }
    else
    {
        throw py::type_error("Cannot assign, function wrapper not working properly.");
    }

};

template <typename T> inline opencascade::handle<T> cast_pyobject(const py::object &arg){
  if(py::isinstance(arg, py::type::of<T>()))
      return opencascade::handle(py::cast<T*>(arg));
  else
      throw py::value_error(fmt::format("Cannot convert the argument to the required type {}", typeid(T).name()));
};

template<typename T, template<typename> typename Deleter = std::default_delete>
struct shared_ptr : public std::shared_ptr<T>{
    explicit shared_ptr(T* t = nullptr) : std::shared_ptr<T>(t, Deleter<T>()) {};
    void reset(T* t = nullptr) { std::shared_ptr<T>::reset(t, Deleter<T>());  };
};

template<typename T> struct nodelete{
    void operator()(T* p) const {};
};

template<typename T> using shared_ptr_nodelete = shared_ptr<T,nodelete>;

PYBIND11_DECLARE_HOLDER_TYPE(T, shared_ptr<T>);
PYBIND11_DECLARE_HOLDER_TYPE(T, shared_ptr_nodelete<T>);

// Phase 1 of the generated bindings (register_<mod>_enums: submodules, enums, class declarations,
// exceptions) stays eager, so that every type exists when a member is defined or a value converted;
// phase 2 (register_<mod>: the members) runs per package on first use, after the packages owning
// its base classes: inherited members are looked up through the Python MRO.
namespace ocp_lazy {

using reg_fn = void (*)(py::module &);

struct desc {            // one row per package, emitted by template_main.j2 in toposort order
    const char *name;    // submodule name ("gp")
    reg_fn pre;          // register_<name>_enums : phase 1, run eagerly by PYBIND11_MODULE
    reg_fn reg;          // register_<name>       : phase 2, run by ensure()
    const char *deps;    // "Standard,gp": packages whose phase 2 must precede this one
};

struct entry {
    const desc *d;
    std::vector<PyObject *> mods;      // package submodule + its namespace submodules (owned)
    std::vector<PyObject *> hidden;    // per mods[i]: its dict as phase 1 left it, while the public names are hidden (owned)
    std::vector<entry *> deps;
    enum { fresh, loading, done, failed } state = fresh;
    std::string error;
};

struct registry {                      // holds Python references as raw pointers: nothing is released at exit
    PyObject *main = nullptr;          // the extension module (OCP.OCP); borrowed
    std::vector<entry> entries;        // reserved once in install(): pointers stay valid
    std::unordered_map<std::string, entry *> by_name;
    std::unordered_map<std::type_index, entry *> by_type;
    int depth = 0;                     // > 0 while a register_<mod> runs
};

inline registry &reg() { static registry r; return r; }

// A dunder is metadata or an import-machinery probe (__path__, __wrapped__ ...) unless phase 2
// defines it: the module-level operators of [Operators] in ocp.toml, or __all__ (star import).
inline bool is_meta(PyObject *name) {
    if (PyUnicode_GET_LENGTH(name) < 2 || PyUnicode_READ_CHAR(name, 0) != '_' || PyUnicode_READ_CHAR(name, 1) != '_') return false;
    for (const char *n : {"__all__", "__mul__", "__rmul__", "__imul__", "__truediv__", "__itruediv__", "__add__",
                          "__iadd__", "__neg__", "__sub__", "__isub__", "__pow__", "__ipow__", "__call__"})
        if (PyUnicode_CompareWithASCIIString(name, n) == 0) return false;
    return true;
}

inline py::dict snapshot(PyObject *mod) {
    py::dict d = py::reinterpret_steal<py::dict>(PyDict_Copy(PyModule_GetDict(mod)));
    if (!d) throw py::error_already_set();
    return d;
}

inline void restore(PyObject *mod, PyObject *dict) {
    PyObject *d = PyModule_GetDict(mod);
    PyDict_Clear(d);
    if (PyDict_Update(d, dict) < 0) throw py::error_already_set();
}

// While a package registers, a cast must not load another package (the only casts are
// default-argument values, never handed to Python) and no garbage collection may run: a
// finalizer could release the GIL (CPython <= 3.11 collects inside the allocating C call)
// and expose the half-defined package to another thread.
struct registering {
    int gc;
    registering() : gc(PyGC_Disable()) { ++reg().depth; }
    ~registering() { --reg().depth; if (gc) PyGC_Enable(); }
};

[[noreturn]] inline void fail(entry &e, const std::vector<py::dict> &before, const char *what) {
    for (std::size_t i = 0; i < e.mods.size(); ++i) restore(e.mods[i], before[i].ptr());   // metadata and the hooks: every public name keeps rethrowing
    e.state = entry::failed;
    e.error = fmt::format("OCP: registration of package '{}' failed: {}", e.d->name, what);
    throw std::runtime_error(e.error);     // same text on the first and on every later attempt
}

inline void ensure(entry &e) {
    switch (e.state) {
    case entry::done:
    case entry::loading:                   // nothing re-enters a running load: deps form a DAG, casts are suppressed
        return;
    case entry::failed:
        throw std::runtime_error(e.error);
    case entry::fresh:
        break;
    }
    std::vector<py::dict> before;
    for (PyObject *mod : e.mods) before.push_back(snapshot(mod));
    e.state = entry::loading;
    registering guard;
    try {
        for (entry *dep : e.deps) ensure(*dep);
        for (std::size_t i = 0; i < e.mods.size(); ++i)   // hooks gone, phase-1 dict back in its original order:
            restore(e.mods[i], e.hidden[i]);               // register_<mod> looks its classes up here
        py::module m = py::reinterpret_borrow<py::module>(reg().main);
        e.d->reg(m);
    } catch (const std::exception &ex) {
        fail(e, before, ex.what());
    } catch (...) {                        // OCCT exceptions are not std::exceptions
        fail(e, before, "non-standard exception");
    }
    for (PyObject *&h : e.hidden) Py_CLEAR(h);   // on failure it stays: it keeps the phase-1 types alive
    e.state = entry::done;
}

inline void load_owner(const std::type_info &t) {
    auto it = reg().by_type.find(t);
    if (it != reg().by_type.end()) ensure(*it->second);
}

inline void on_cast(const std::type_info &static_type, const std::type_info *dynamic_type) {
    if (reg().depth > 0 || reg().by_type.empty()) return;   // registering, or OCP_LAZY=0
    load_owner(static_type);
    if (dynamic_type && *dynamic_type != static_type) load_owner(*dynamic_type);
}

inline void hide(entry &e, PyObject *mod) {
    Py_INCREF(mod);
    e.mods.push_back(mod);
    PyObject *d = PyModule_GetDict(mod);
    PyObject *hidden = snapshot(mod).release().ptr();
    e.hidden.push_back(hidden);
    std::vector<PyObject *> nested;
    Py_ssize_t pos = 0;
    PyObject *k, *v;
    while (PyDict_Next(hidden, &pos, &k, &v)) {
        if (is_meta(k)) continue;          // metadata (__name__, __doc__, __spec__ ...) stays visible
        if (PyModule_Check(v)) nested.push_back(v);
        else if (PyType_Check(v))
            if (auto *ti = py::detail::get_type_info(reinterpret_cast<PyTypeObject *>(v))) reg().by_type.emplace(*ti->cpptype, &e);
        PyDict_DelItem(d, k);
    }
    std::string modname = PyModule_GetName(mod);
    py::cpp_function getattr_hook([&e, mod, modname](py::str name) -> py::object {
        if (is_meta(name.ptr())) {         // import-machinery probes (__path__ ...) must not load
            PyErr_Format(PyExc_AttributeError, "module '%s' has no attribute '%U'", modname.c_str(), name.ptr());
            throw py::error_already_set();
        }
        ensure(e);                                              // hooks removed, or the failure rethrown
        PyObject *res = PyObject_GetAttr(mod, name.ptr());      // plain lookup now: standard error text;
        if (!res) throw py::error_already_set();                // for __all__ the AttributeError makes
        return py::reinterpret_steal<py::object>(res);          // `import *` fall back to __dict__
    });
    py::cpp_function dir_hook([&e, mod]() -> py::object {
        ensure(e);
        PyObject *res = PyObject_Dir(mod);
        if (!res) throw py::error_already_set();
        return py::reinterpret_steal<py::object>(res);
    });
    // CPython calls both through a borrowed reference and ensure() drops them from the
    // dict while they run: the references released here are kept on purpose.
    PyDict_SetItemString(d, "__getattr__", getattr_hook.release().ptr());
    PyDict_SetItemString(d, "__dir__", dir_hook.release().ptr());
    for (PyObject *n : nested) hide(e, n);
}

// Called once at the end of PYBIND11_MODULE, after phase 1 of every row.
template <std::size_t N>
inline void install(py::module &main, const desc (&table)[N]) {
    registry &r = reg();
    r.main = main.ptr();
    r.entries.reserve(N);
    for (const desc &d : table) {
        r.entries.push_back({&d});
        r.by_name[d.name] = &r.entries.back();
    }
    for (entry &e : r.entries) {
        std::istringstream in(e.d->deps);
        for (std::string dep; std::getline(in, dep, ',');) {
            auto it = r.by_name.find(dep);
            if (it == r.by_name.end())
                throw std::runtime_error(fmt::format("OCP: package '{}' depends on unknown package '{}'", e.d->name, dep));
            e.deps.push_back(it->second);
        }
    }
    main.def("_load", [](const std::string &name) {
        auto it = reg().by_name.find(name);
        if (it == reg().by_name.end()) throw py::key_error(name);
        ensure(*it->second);
    }, py::arg("name"), "Register the members of one OCP package now (idempotent).");
    main.def("_load_all", []() { for (entry &e : reg().entries) ensure(e); },
             "Register the members of every OCP package now (idempotent).");

    if (py::module::import("os").attr("environ").attr("get")("OCP_LAZY", "").cast<std::string>() == "0") {   // eager: exactly the former PyInit, no hooks ever installed
        for (entry &e : r.entries) { e.d->reg(main); e.state = entry::done; }
        return;
    }
    for (entry &e : r.entries) hide(e, main.attr(e.d->name).ptr());
}

} // namespace ocp_lazy

// Every C++ -> Python cast of a class instance passes here first (type_caster_base::src_and_type);
// this file is the include_pre of every generated TU, so the specialisation is the same in all of them.
namespace pybind11 {
template <typename T>
struct polymorphic_type_hook<T, void> {
    static const void *get(const T *src, const std::type_info *&type) {
        const void *res = polymorphic_type_hook_base<T>::get(src, type);
        if (src) ocp_lazy::on_cast(typeid(T), type);
        return res;
    }
};
} // namespace pybind11

#include "pystreambuf.h"
