//===-------- FineIBTHashesSection - Creates .ibt.fine.plt section --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The FineIBT final DSO can contain multiple functions, which are provenient
// from different object files. During the Linking of these objects, the PLT
// entries generated require the hash of each function. Given that these
// functions are accessed through the PLT, they are not present in any of the
// object files being linked. Also, as these hashes are based on the prototype
// respective to each function, and that such signature information is not
// available in the binary objects, the linker needs to be actively informed
// about such hashes.
//
// This pass creates functions which will hold the FineIBT hashes so they can
// be accessed by the linker and used during PLT emission. These functions are
// placed in a special section ".ibt.fine.plt" so they can all be discarded
// once the final DSO is fully linked.
//
//===----------------------------------------------------------------------===//

// TODO: fix includes
//#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Target/TargetMachine.h"
#include <iostream>

using namespace llvm;

#define DEBUG_TYPE "FineIBTHashes"

namespace {

class FineIBTHashesSection : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  FineIBTHashesSection() : ModulePass(ID) {
    initializeFineIBTHashesSectionPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
  }

  bool runOnModule(Module &M) override;

private:
};
} // namespace

char FineIBTHashesSection::ID = 0;

INITIALIZE_PASS(FineIBTHashesSection, "FineIBTHashes",
                "Create a new section in the object with FineIBT hash entries",
                false, false)

ModulePass *llvm::createFineIBTHashesSectionPass() {
  return new FineIBTHashesSection();
}

// Create .ibt.fine.plt entries for the hashes later used to emit the PLT.
static void CreatePLTTemplate(Module &M, Function *F) {
  std::string N;
  StringRef name = F->getName();
  if (F->isIntrinsic()) {
    if (name.startswith("llvm.memcpy"))
      N = "__ibt_fine_plt_memcpy";
    else if (name.startswith("llvm.memset"))
      N = "__ibt_fine_plt_memset";
    else if (name.startswith("llvm.memmove"))
      N = "__ibt_fine_plt_memmove";
  } else {
    N = "__ibt_fine_plt_" + name.str();
  }

  LLVMContext &Ctx = M.getContext();
  auto Ty = F->getFunctionType();
  Function *Fn = Function::Create(Ty, GlobalValue::WeakAnyLinkage, N, &M);
  Fn->setDoesNotReturn();
  Fn->setVisibility(GlobalValue::DefaultVisibility);
  Fn->setCallingConv(CallingConv::C);
  Fn->setSection(".ibt.fine.plt");
  Fn->setIsMaterializable(true);

  AttrBuilder B;
  B.addAttribute(llvm::Attribute::NoUnwind);
  B.addAttribute(llvm::Attribute::Naked);
  Fn->addAttributes(llvm::AttributeList::FunctionIndex, B);

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", Fn);
  IRBuilder<> Builder(Entry);
  Builder.CreateUnreachable();
}

// some intrinsics result in calls through the PLT, thus need a FineIBT hash.
static bool IsExternalIntrinsic(Function *F) {
  StringRef name = F->getName();
  if (name.startswith("llvm.memcpy"))
    return true;
  if (name.startswith("llvm.memset"))
    return true;
  if (name.startswith("llvm.memmove"))
    return true;
  return false;
}

bool FineIBTHashesSection::runOnModule(Module &M) {
  if (!M.getModuleFlag("cf-protection-fine"))
    return false;

  const TargetMachine *TM =
      &getAnalysis<TargetPassConfig>().getTM<TargetMachine>();
  if (!TM)
    return false;

  for (auto &F : M) {
    if (F.getName().startswith("__ibt_fine_plt"))
      continue;
    // most intrinsics, static and no/coarse cf checks don't need special PLT
    // entries.
    if (!F.doesNoCfCheck() && !F.doesCoarseCfCheck() && !F.hasLocalLinkage()) {
      if (F.isIntrinsic() && !IsExternalIntrinsic(&F))
        continue;
      CreatePLTTemplate(M, &F);
    }
  }

  return true;
}
