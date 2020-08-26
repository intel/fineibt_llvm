; RUN: llc -mtriple=x86_64-unknown-unknown -x86-indirect-branch-tracking < %s | FileCheck %s

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; This test verifies the handling of ''coarsecf_check'' attribute by the    ;;
;; backend. The file was generated using the following C code:               ;;
;;                                                                           ;;
;; void __attribute__((coarsecf_check)) CoarseCfCheckFunc(void) {}           ;;
;;                                                                           ;;
;; typedef void(*FuncPointer)(void);                                         ;;
;; void CoarseCfCheckCall(FuncPointer f) {                                   ;;
;;   __attribute__((coarsecf_check)) FuncPointer p = f;                      ;;
;;   (*p)();                                                                 ;;
;; }                                                                         ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; Make sure that a function with ''coarsecf_check'' attribute is only
; instrumented endbr instruction at the beginning (no FineIBT checks).
define void @CoarseCfCheckFunc() #0 {
; CHECK-LABEL: CoarseCfCheckFunc
; CHECK:       endbr64
; CHECK-NOT:   {{xorl.*r11}}
; CHECK-NOT:   je
; CHECK-NOT:   hlt
; CHECK-NOT:   int3
; CHECK-NOT:   int3
; CHECK:       retq
entry:
  ret void
}

; Make sure that FineIBT hash set was not placed before the call.
define void @CoarseCfCheckCall(void ()* %f) {
; CHECK-LABEL:   CoarseCfCheckCall
; CHECK-NOT:     {{mov.*r11}}
; CHECK:         call
entry:
	%f.addr = alloca void ()*, align 4
	%p = alloca void()*, align 4
	store void ()* %f, void ()** %f.addr, align 4
  %0 = load void ()*, void ()** %f.addr, align 4
  store void ()* %0, void ()** %p, align 4
  %1 = load void ()*, void ()** %p, align 4
  call void %1() #1
  ret void
}

attributes #0 = { noinline coarsecf_check nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { coarsecf_check }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 4, !"cf-protection-branch", i32 1}
!1 = !{i32 4, !"cf-protection-fine", i32 1}
