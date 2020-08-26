; RUN: llc -mtriple=x86_64-unknown-unknown < %s | FileCheck %s --check-prefix=FINE
; RUN: llc -mtriple=x86_64-unknown-unknown < %s | FileCheck %s --check-prefix=ITAKEN
; RUN: llc -mtriple=x86_64-unknown-unknown < %s | FileCheck %s --check-prefix=INTAKEN


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Test1
;; -----
;; Checks ENDBR + FineIBT checks insertion in case of indirect call
;; instruction. These should be added to the called function (test2)
;; although it is internal.
;; A FineIBT hash set instruction should be added before the indirect
;; call. Also since the function is not internal, ENDBR and FineIBT
;; checks should be added to its first basic block.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
define void @test1() {
;FINE-LABEL:   test1:
;FINE:         endbr64
;FINE-NEXT:    {{xor.*r11}}
;FINE-NEXT:    je
;FINE:         hlt
;FINE-NEXT:    int3
;FINE-NEXT:    int3
;FINE:         {{mov.*r11}}
;FINE-NEXT:    call
entry:
  %f = alloca i32 (...)*, align 8
  store i32 (...)* bitcast (i32 (i32)* @test2 to i32 (...)*), i32 (...)** %f, align 8
  %0 = load i32 (...)*, i32 (...)** %f, align 8
  %call = call i32 (...) %0()
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Test2
;; -----
;; Checks ENDBR + FineIBT checksinsertion in case of function that is
;; address taken.  Since the function's address was taken by test1() and
;; despite being internal, check for added ENDBR and FineIBT checks at
;; the beginning of the function.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
define internal i32 @test2(i32 %a) {
;ITAKEN-LABEL:   test2:
;ITAKEN:         endbr64
;ITAKEN-NEXT:    {{xorl .*r11}}
;ITAKEN-NEXT:    je
;ITAKEN:         hlt
;ITAKEN-NEXT:    int3
;ITAKEN-NEXT:    int3
  ret i32 1
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Test3
;; -----
;; Checks ENDBR + FineIBT insertion in case of internal function.
;; Since the function is internal and its address was not taken,
;; make sure that ENDBR was not added at the beginning of the
;; function.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
define internal i8 @test3(){
;INTAKEN-LABEL:   test3
;INTAKEN-NOT:     endbr64
  ret i8 1
}

!llvm.module.flags = !{!0, !1}

!0 = !{i32 4, !"cf-protection-branch", i32 1}
!1 = !{i32 4, !"cf-protection-fine", i32 1}
