<*
 The shader `src/main.c3` actually runs, compiled and validated on its own.

 This imports `kong::SHADER` rather than keeping a copy, so it cannot pass
 against a stale duplicate while the real shader is broken.
*>
module main_shader_test;

import std::io;
import std::io::path;
import kong;
import shady;

fn void the_cube_shader_compiles() @test
{
	shady::Diagnostic diagnostic;
	char[]? spirv = shady::compile(kong::SHADER, &diagnostic);
	if (catch spirv)
	{
		io::printfn("shady error at %d:%d - %s", diagnostic.line, diagnostic.column, diagnostic.message);
		unreachable("the cube shader should compile");
	}
	defer free(spirv);

	assert(((uint*)spirv.ptr)[0] == shady::MAGIC, "bad magic");

	path::mkdir("build/shady", recursive: true)!!;
	file::save("build/shady/main.spirv", spirv)!!;
	io::printfn("main.c3 shader: %d bytes of SPIR-V", spirv.len);
}
