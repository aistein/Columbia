; ModuleID = 'int_to_float.c'
source_filename = "int_to_float.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin16.5.0"

; Function Attrs: nounwind
define i32 @main() #0 {
  %retval = alloca i32, align 4
  %x = alloca i32, align 4
  %y = alloca float, align 4
  %z = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  store i32 1, i32* %x, align 4
  store float 0x4000CCCCC0000000, float* %y, align 4
  %1 = load i32, i32* %x, align 4
  %conv = sitofp i32 %1 to float
  %2 = load float, float* %y, align 4
  %add = fadd float %conv, %2
  %conv1 = fptosi float %add to i32
  store i32 %conv1, i32* %z, align 4
  ret i32 0
}

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "target-features"="+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"Apple LLVM version 8.1.0 (clang-802.0.38)"}
