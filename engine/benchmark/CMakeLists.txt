cpmaddpackage(NAME tabulate GITHUB_REPOSITORY p-ranav/tabulate VERSION 1.5)
cpmaddpackage(
  NAME
  ftxui
  GITHUB_REPOSITORY
  ArthurSonzogni/FTXUI
  VERSION
  5.0.0
  OPTIONS
  "FTXUI_BUILD_EXAMPLES OFF"
  "FTXUI_BUILD_DOCS OFF"
  "FTXUI_BUILD_TESTS OFF")

find_package(Arrow CONFIG REQUIRED)
find_package(Parquet CONFIG REQUIRED)
add_executable(kribu_benchmark_main benchmark_main.cpp)
target_link_libraries(
  kribu_benchmark_main
  PRIVATE kribu_engine
          Arrow::arrow_shared
          Parquet::parquet_shared
          tabulate
          ftxui::screen
          ftxui::dom
          ftxui::component)
