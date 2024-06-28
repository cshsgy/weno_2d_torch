macro(add_shader namel)
  set(SHADER_SOURCE ${CMAKE_SOURCE_DIR}/${namel}.metal)
  set(SHADER_LIBRARY ${CMAKE_BINARY_DIR}/${namel}.metallib)
  string(TOUPPER ${namel} nameu)

  add_custom_command(
      OUTPUT ${SHADER_LIBRARY}
      COMMAND xcrun -sdk macosx metal -c ${SHADER_SOURCE} -o ${CMAKE_BINARY_DIR}/${namel}.air
      COMMAND xcrun -sdk macosx metallib ${CMAKE_BINARY_DIR}/${namel}.air -o ${SHADER_LIBRARY}
      COMMAND rm ${CMAKE_BINARY_DIR}/${namel}.air
      DEPENDS ${SHADER_SOURCE}
      COMMENT "Compiling ${namel} shader to ${namel}.metallib"
  )

  add_custom_target(
      ${nameu} ALL
      DEPENDS ${SHADER_LIBRARY}
  )
endmacro()
