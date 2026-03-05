<div align="right">
Version 4.2.0
</div>
This is an empty project that can be employed as a skeleton for any
hiperlife-based project.

## Description
A hiperlife's project is composed by configuration files and
applications:
```bash
.
├── CMakeLists.txt       # <--- Main configuration file. There is no need to modify it
├── userConfig.cmake     # <--- User's configuration file. Change the name of the project and modify/add/remove applications
├── App1                 # <--- Application called "App1"
|   ├── CMakeLists.txt   # <------ Application config file. There is no need to modify it
|   ├── App1.cpp         # <------ Source file
|   ├── AuxApp1.h        # <------ Source file
|   └── AuxApp1.cpp      # <------ Source file
├── App2                 # <--- Application called "App2"
|   ├── CMakeLists.txt   # <------ Application config file. There is no need to modify it
|   ├── App2.cpp         # <------ Source file
|   ├── AuxApp2.h        # <------ Source file
|   ├── AuxApp2.cpp      # <------ Source file
|   └── ...              # <------ Other source files
└── ...                  # <--- Other applications
```
Every project has a main configuration file *CMakeLists.txt* and a user's configuration file *userConfig.cmake*. The former contains the instructions to configure the project (see [Configure, compile and install a project](#Configure-compile-and-install-a-project)) and there is no need to modify it. The later contains the user configuration of the project and may be modified by the user to change the name of the project and to modify, add or remove applications (see [How to modify a project](#how-to-modify-a-project)).

Each application is contained in a folder that gives the name to the application, as in *App1* or *App2*. Every application needs a configuration file, *CMakeLists.txt*, that does not need to be modified and the source files of the application, i.e. *App1.cpp*, *AuxApp1.h* and *AuxApp1.cpp*. In particular, this project contains a folder, *EmptyApp*, which exemplifies how an application in the project may be added and structured. A main file called *EmptyApp.cpp* contains the `main()` function, and is
complemented with auxiliary functions that may be declared and defined in *AuxEmptyApp.h* and *AuxEmptyApp.cpp*, respectively.


## Configure, compile and install a project
To configure the project go into the build folder (`cd <path/to/hl_BaseProject>/build`) and configure using `cmake` as follows:
```bash
cmake \
      -DHPLFE_BASE_PATH=<path/to/hiperlife> \
      -DCMAKE_INSTALL_PREFIX=<path/to/installation> \
      -DCMAKE_BUILD_TYPE=<RELEASE/debug> \
      ..
```
where you have to replace `<path/to/XX>` with the corresponding paths and choose between `release` (default) and `debug` compilation types.

Once configured, you can compile and install the project with `make install` or `make -jX install` to compile using several processors, where `X` stands for the number of processors.


## How to modify a project

### Change the name of the project
Open the file *userConfig.cmake* and replace *hl_BaseProject* with the new name of your project *MyProject*:
```cmake
set(PROJECT_NAME MyProject)
```

### Modify an existing application

You may modify the name of your application by changing the name of the folder and its corresponding name in *userConfig.cmake*. Open the file *userConfig.cmake* and replace the previous name of your application, for instance, *EmptyApp*, with the new name of your application *MyApp*:
 ```cmake
list(APPEND APPS MyApp)
```

You may modify the names of the source files at convenience. Also, you may add or remove auxiliary files in the folder as you need. The *CMakeLists.txt* inside the application's folder will take all them into account to build your application.

### Add a new application

If you want to add another application to the project, say *MyNewApp*, generate a folder with the name *MyNewApp*, and copy the *CMakeLists.txt* from the *EmptyApp* folder into *MyNewApp* folder (`cp EmptyApp/CMakeLists.txt MyNewApp/`). Now you may create a main source (.cpp) file in `MyNewApp/` and as many auxiliary files as you may need for your application.

You will also need to add another line to the *userConfig.cmake* file so that your application is configured. If your application does not need vtk nor gmsh, right after
```cmake
list(APPEND APPS EmptyApp)
```
add the line
```cmake
list(APPEND APPS MyNewApp)
```
 If your app needs both vtk and gmsh libraries, add:
```cmake
list(APPEND APPS_VTK_GMSH MyNewApp)
```
If your app only needs vtk, add:
```cmake
list(APPEND APPS_VTK MyNewApp)
```
If your app only needs gmsh, add:
```cmake
list(APPEND APPS_GMSH MyNewApp)
```
