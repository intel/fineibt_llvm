//===---- X86IndirectBranchTracking.cpp - Enables CET IBT mechanism -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a pass that enables Indirect Branch Tracking (IBT) as part
// of Control-Flow Enforcement Technology (CET).
// The pass adds ENDBR (End Branch) machine instructions at the beginning of
// each basic block or function that is referenced by an indrect jump/call
// instruction.
// The ENDBR instructions have a NOP encoding and as such are ignored in
// targets that do not support CET IBT mechanism.
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "X86TargetMachine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/MC/MCSymbolELF.h"
#include <iostream>

using namespace llvm;

#define DEBUG_TYPE "x86-indirect-branch-tracking"
#define wwarn WithColor::warning() << "FineIBT: "

static cl::opt<bool> IndirectBranchTracking(
    "x86-indirect-branch-tracking", cl::init(false), cl::Hidden,
    cl::desc("Enable X86 indirect branch tracking pass."));

STATISTIC(NumEndBranchAdded, "Number of ENDBR instructions added");

namespace {
class X86IndirectBranchTrackingPass : public MachineFunctionPass {
public:
  X86IndirectBranchTrackingPass() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "X86 Indirect Branch Tracking";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  static char ID;

  /// Machine instruction info used throughout the class.
  const X86InstrInfo *TII = nullptr;

  /// Endbr opcode for the current machine function.
  unsigned int EndbrOpcode = 0;

  /// Adds a new ENDBR instruction to the beginning of the MBB.
  /// The function will not add it if already exists.
  /// It will add ENDBR32 or ENDBR64 opcode, depending on the target.
  /// \returns true if the ENDBR was added and false otherwise.
  bool addENDBR(MachineBasicBlock &MBB, MachineBasicBlock::iterator I) const;

  /// Add endbr instruction as the first instruction in functions that can be
  /// reached through indirect calls. This is a coarse-grained IBT scheme.
  bool applyCoarseIBT(MachineFunction &MF);

  /// Add endbr + FineIBT checking as first instruction in functions that can be
  /// creached through indirect calls. This is a fine-grained IBT scheme.
  bool applyFineIBT(MachineFunction &MF);
};
} // end anonymous namespace

char X86IndirectBranchTrackingPass::ID = 0;

FunctionPass *llvm::createX86IndirectBranchTrackingPass() {
  return new X86IndirectBranchTrackingPass();
}

bool X86IndirectBranchTrackingPass::addENDBR(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator I) const {
  assert(TII && "Target instruction info was not initialized");
  assert((X86::ENDBR64 == EndbrOpcode || X86::ENDBR32 == EndbrOpcode) &&
         "Unexpected Endbr opcode");

  // If the MBB/I is empty or the current instruction is not ENDBR,
  // insert ENDBR instruction to the location of I.
  if (I == MBB.end() || I->getOpcode() != EndbrOpcode) {
    BuildMI(MBB, I, MBB.findDebugLoc(I), TII->get(EndbrOpcode));
    ++NumEndBranchAdded;
    return true;
  }
  return false;
}

static bool IsCallReturnTwice(llvm::MachineOperand &MOp) {
  if (!MOp.isGlobal())
    return false;
  auto *CalleeFn = dyn_cast<Function>(MOp.getGlobal());
  if (!CalleeFn)
    return false;
  AttributeList Attrs = CalleeFn->getAttributes();
  return Attrs.hasFnAttribute(Attribute::ReturnsTwice);
}

// main is called from libc through an opaque pointer, thus always coarse.
static bool isAlwaysCoarse(StringRef name) {
  if (name.equals("main"))
    return true;
  return false;
}

static bool IsWeakAliasTarget(const Function *F) {
  const Module *M = F->getParent();
  for (auto &A : M->aliases()) {
    if (A.getAliasee() == F && A.hasWeakLinkage())
      return true;
  }
  return false;
}

// this is just a debug method to help spotting new symbol corner cases
static bool verifySymbol(StringRef name) {
  if (name.equals("memcpy"))
    return true;
  if (name.equals("memset"))
    return true;
  if (name.equals("memmove"))
    return true;
  LLVM_DEBUG(wwarn << "Handling unknown symbol: " << name << "\n");
  return false;
}

static bool FixDCalls(MachineFunction &MF) {
  bool Changed = false;

  for (auto &BB : MF) {
    for (auto &I : BB) {
      unsigned Opcode = I.getOpcode();
      if (Opcode == X86::CALL64pcrel32 || Opcode == X86::TAILJMPd ||
          Opcode == X86::TAILJMPd64_CC || Opcode == X86::TAILJMPd_CC ||
          Opcode == X86::TAILJMPd64) {

        auto &O = I.getOperand(0);
        if (O.getOffset()) {
          LLVM_DEBUG(StringRef name = MF.getName();
                     wwarn << "Skipping call already offset'ed in: " << name
                           << "\n";);
          continue;
        }

        if (O.isSymbol()) {
          verifySymbol(O.getSymbolName());
          O.setOffset(32);
          continue;
        }

        if (O.isMCSymbol()) {
          verifySymbol(O.getMCSymbol()->getName());
          O.setOffset(32);
          continue;
        }

        if (O.isGlobal()) {
          const Value *Target = O.getGlobal();
          const GlobalAlias *GAlias = dyn_cast_or_null<GlobalAlias>(Target);
          if (GAlias && GAlias->hasWeakLinkage()) {
            // Targets of weak aliases are always padded.
            O.setOffset(32);
          } else {
            // If not a weak alias, we should be able to identify the call
            // target accurately. Check if it is static, address-traken or
            // a weakalias target and set the call offset accordingly.
            Target = Target->stripPointerCastsAndAliases();
            if (!Target) {
              LLVM_DEBUG(StringRef name = MF.getName();
                         wwarn << "Unknown alias target in " << name
                               << " (skip).\n";);
              continue;
            }
            const Function *F = dyn_cast_or_null<Function>(Target);
            if (!F) {
              LLVM_DEBUG(StringRef name = MF.getName();
                         wwarn << "Unknown alias target in " << name
                               << " (skip).\n";);
              continue;
            }
            // weak alias targets must have a 32b padding on their prologue
            // even if they are static, because they can be replaced during
            // linking by instrumented versions of the function.
            if (!F->hasAddressTaken() && F->hasLocalLinkage() &&
                !IsWeakAliasTarget(F)) {
              LLVM_DEBUG(wwarn << F->getName()
                               << " is not IBT instrumented (skip).\n");
              continue;
            }
            O.setOffset(32);
          }
        }
      }
    }
  }
  return Changed;
}

static bool FixICalls(MachineFunction &MF) {
  bool Changed = false;
  RegScavenger RS;
  unsigned AuxReg;

  const X86Subtarget &SubTarget = MF.getSubtarget<X86Subtarget>();
  auto TII = SubTarget.getInstrInfo();

  for (auto &BB : MF) {
    for (auto &I : BB) {
      unsigned Opcode = I.getOpcode();

      if (Opcode != X86::CALL64r && Opcode != X86::CALL64m &&
          // Fine jumps are not supported...
          // Opcode != X86::JMP64r && Opcode != X86::JMP64m &&
          // Opcode != X86::JMP32r && Opcode != X86::JMP32m &&
          // Opcode != X86::JMP16r && Opcode != X86::JMP16m &&
          Opcode != X86::TAILJMPr64 && Opcode != X86::TAILJMPr &&
          Opcode != X86::TAILJMPm64 && Opcode != X86::TAILJMPm &&
          Opcode != X86::TAILJMPm64_REX && Opcode != X86::TAILJMPr64_REX)
        continue;

      // Instructions with attribute CoarseCfCheck have Hash = 0. Skip it.
      if (I.getPrototypeHash() == 0) {
        LLVM_DEBUG(WithColor::warning()
                     << "FineIBT: NULL Hash in " << MF.getName() << "\n");
      }
      // if R11 is used as a pointer, we need to use a different register.
      MachineOperand &MO = I.getOperand(0);
      if (MO.isReg() && MO.getReg() == X86::R11) {
        RS.enterBasicBlock(BB);
        RS.forward(I);
        AuxReg = RS.FindUnusedReg(&X86::GR64RegClass);
        if (!AuxReg) {
          // TODO: this case needs to be fixed with register scavenging.
          WithColor::warning()
            << "FineIBT: No register available in " << MF.getName() << ".\n";
          continue;
        }
        MO.setReg(AuxReg);
        BuildMI(BB, I, DebugLoc(), TII->get(X86::MOV64rr), AuxReg)
          .addReg(X86::R11);
      }

      // for CALL64m/TAILJMPm we need to also check the second register
      if (Opcode == X86::CALL64m || Opcode == X86::TAILJMPm64)
      {
        MachineOperand &MO = I.getOperand(2);
        if (MO.isReg() && MO.getReg() == X86::R11) {
          RegScavenger RS;
          RS.enterBasicBlock(BB);
          RS.forward(I);
          AuxReg = RS.FindUnusedReg(&X86::GR64RegClass);
          if (!AuxReg) {
            WithColor::warning()
              << "FineIBT: No register available in " << MF.getName() << ".\n";
            continue;
          }
          MO.setReg(AuxReg);
          BuildMI(BB, I, DebugLoc(), TII->get(X86::MOV64rr), AuxReg)
            .addReg(X86::R11);
        }
      }
      Changed = true;
      BuildMI(BB, I, DebugLoc(), TII->get(X86::MOV64ri), X86::R11)
        .addImm(I.getPrototypeHash());
    }
  }
  return Changed;
}

// Large code model, non-internal function or function whose address
// was taken, can be accessed through indirect calls. Mark the first
// BB with ENDBR instruction unless nocf_check attribute is used.
bool X86IndirectBranchTrackingPass::applyCoarseIBT(MachineFunction &MF) {
  bool Changed = false;
  const X86TargetMachine *TM =
      static_cast<const X86TargetMachine *>(&MF.getTarget());

  if ((TM->getCodeModel() == CodeModel::Large ||
       MF.getFunction().hasAddressTaken() ||
       !MF.getFunction().hasLocalLinkage()) &&
      !MF.getFunction().doesNoCfCheck()) {

    auto MBB = MF.begin();
    Changed |= addENDBR(*MBB, MBB->begin());
  }
  return Changed;
}

// Add endbr instruction as the first instruction of functions that can be
// accessed through indirect calls.
bool X86IndirectBranchTrackingPass::applyFineIBT(MachineFunction &MF) {
  // this is a regular Fine IBT target function. add the Fine IBT check
  // and mark the direct call bypass.

  Function &F = MF.getFunction();
  // .ibt.fine.plt entries hold the function hash as an info for the linker.
  // This is a workaround to prevent the dependency on LTO, and it is harmless
  // since the linker discards the .ibt.fine.plt section anyway.
  if (F.getName().startswith("__ibt_fine_plt")) {
    auto MBB = MF.begin();
    auto I = MBB->begin();
    BuildMI(*MBB, I, MBB->findDebugLoc(I), TII->get(X86::XOR32ri), X86::R11D)
        .addReg(X86::R11D, RegState::Undef)
        .addImm(F.getFunctionType()->getPrototypeHash());

    // int3 to prevent nasty shenanigans.
    BuildMI(*MBB, I, MBB->findDebugLoc(I), TII->get(X86::INT3));

    // 8 bytes nop.
    BuildMI(*MBB, I, MBB->findDebugLoc(I), TII->get(X86::NOOPL))
        .addReg(X86::RAX)
        .addImm(1)
        .addReg(X86::RAX)
        .addImm(512)
        .addReg(0);

    // 9 bytes nop.
    BuildMI(*MBB, I, MBB->findDebugLoc(I), TII->get(X86::NOOPW))
        .addReg(X86::RAX)
        .addImm(1)
        .addReg(X86::RAX)
        .addImm(512)
        .addReg(0);

    // 10 bytes nop.
    BuildMI(*MBB, I, MBB->findDebugLoc(I), TII->get(X86::NOOPW))
        .addReg(X86::RAX)
        .addImm(1)
        .addReg(X86::RAX)
        .addImm(512)
        .addReg(X86::FS);

    return true;
  }

  // Static functions targeted by weak aliases need to have a nop prologue,
  // because respective calls will have an offset as we don't know if the
  // possible weak replacement candidates have fine-IBT instrumentation.
  if (!F.hasAddressTaken() && F.hasLocalLinkage()) {
    if (IsWeakAliasTarget(&F)) {
      auto MBB = MF.begin();
      auto I = MBB->begin();

    // 8 bytes nop.
    BuildMI(*MBB, I, MBB->findDebugLoc(I), TII->get(X86::NOOPL))
        .addReg(X86::RAX)
        .addImm(1)
        .addReg(X86::RAX)
        .addImm(512)
        .addReg(0);

    // 8 bytes nop.
    BuildMI(*MBB, I, MBB->findDebugLoc(I), TII->get(X86::NOOPL))
        .addReg(X86::RAX)
        .addImm(1)
        .addReg(X86::RAX)
        .addImm(512)
        .addReg(0);

    // 8 bytes nop.
    BuildMI(*MBB, I, MBB->findDebugLoc(I), TII->get(X86::NOOPL))
        .addReg(X86::RAX)
        .addImm(1)
        .addReg(X86::RAX)
        .addImm(512)
        .addReg(0);

    // 8 bytes nop.
    BuildMI(*MBB, I, MBB->findDebugLoc(I), TII->get(X86::NOOPL))
        .addReg(X86::RAX)
        .addImm(1)
        .addReg(X86::RAX)
        .addImm(512)
        .addReg(0);
    }
    return true;
  }

  if (MF.getFunction().doesCoarseCfCheck() || isAlwaysCoarse(MF.getName())) {
    auto MBB = MF.begin();
    auto I = MBB->begin();

    // 4 byte endbr.
    BuildMI(*MBB, I, MBB->findDebugLoc(I), TII->get(X86::ENDBR64));

    // 8 bytes nop.
    BuildMI(*MBB, I, MBB->findDebugLoc(I), TII->get(X86::NOOPL))
        .addReg(X86::RAX)
        .addImm(1)
        .addReg(X86::RAX)
        .addImm(512)
        .addReg(0);

    // 10 bytes nop.
    BuildMI(*MBB, I, MBB->findDebugLoc(I), TII->get(X86::NOOPW))
        .addReg(X86::RAX)
        .addImm(1)
        .addReg(X86::RAX)
        .addImm(512)
        .addReg(X86::FS);

    // 10 bytes nop.
    BuildMI(*MBB, I, MBB->findDebugLoc(I), TII->get(X86::NOOPW))
        .addReg(X86::RAX)
        .addImm(1)
        .addReg(X86::RAX)
        .addImm(512)
        .addReg(X86::FS);

    ++NumEndBranchAdded;

    return true;
  }

  // Get the function's entry block
  MachineBasicBlock *Entry = &MF.front();
  // Create and organize new basic blocks
  // "ChkMBB" will hold the ENDBR + Hash Checks
  // "BitMBB" will hold the FineIBT bit check
  MachineBasicBlock *ChkMBB = MF.CreateMachineBasicBlock();
  MachineBasicBlock *BitMBB = MF.CreateMachineBasicBlock();
  MachineBasicBlock *VltMBB = MF.CreateMachineBasicBlock();
  MF.push_front(VltMBB);
  MF.push_front(BitMBB);
  MF.push_front(ChkMBB);

  ChkMBB->addSuccessor(Entry);
  ChkMBB->addSuccessor(BitMBB);
  BitMBB->addSuccessor(VltMBB);
  ChkMBB->addLiveIn(X86::R11);
  for (const auto &LI : Entry->liveins()) {
    ChkMBB->addLiveIn(LI);
  }
  BitMBB->addLiveIn(X86::R11);
  for (const auto &LI : Entry->liveins()) {
    BitMBB->addLiveIn(LI);
  }

  // FineIBT does not support 32b, disconsider ENDBR32
  BuildMI(ChkMBB, DebugLoc(), TII->get(X86::ENDBR64));
  ++NumEndBranchAdded;

  // Add the hash check
  uint32_t Hash = F.getFunctionType()->getPrototypeHash();
  BuildMI(ChkMBB, DebugLoc(), TII->get(X86::XOR32ri), X86::R11D)
      .addReg(X86::R11D, RegState::Undef)
      .addImm(Hash);

  // Add the JE if the check matches.
  // Use JE as it allows a 1b jmp over the HLT, resulting in a shorter snippet.
  auto MI = BuildMI(ChkMBB, DebugLoc(), TII->get(X86::JCC_1))
      .addMBB(Entry)
      .addImm(X86::COND_E);
  MI->setDoNotRelax(true);

  MI = BuildMI(BitMBB, DebugLoc(), TII->get(X86::TEST8mi))
      .addReg(0)
      .addImm(1)
      .addReg(0)
      .addImm(0x48)
      .addReg(X86::FS)
      .addImm(0x10); // check if the binary has FineIBT

  MI = BuildMI(BitMBB, DebugLoc(), TII->get(X86::JCC_1))
      .addMBB(Entry)
      .addImm(X86::COND_E);
  MI->setDoNotRelax(true);

  // If the check fails,
  BuildMI(VltMBB, DebugLoc(), TII->get(X86::HLT));
  // and catch fire.

  // Ensure 16 byte align (int3 + 6 byte nop)
  BuildMI(VltMBB, DebugLoc(), TII->get(X86::INT3));
  BuildMI(VltMBB, DebugLoc(), TII->get(X86::NOOPW))
      .addReg(X86::RAX)
      .addImm(1)
      .addReg(X86::RAX)
      .addImm(8)
      .addReg(0);

  return true;
}

bool X86IndirectBranchTrackingPass::runOnMachineFunction(MachineFunction &MF) {
  const X86Subtarget &SubTarget = MF.getSubtarget<X86Subtarget>();
  auto &MMI = MF.getMMI();

  // Check that the cf-protection-branch is enabled.
  Metadata *isCFProtectionSupported =
      MMI.getModule()->getModuleFlag("cf-protection-branch");

  Metadata *FineIBT = MMI.getModule()->getModuleFlag("cf-protection-fine");

  // NB: We need to enable IBT in jitted code if JIT compiler is CET enabled.
  const X86TargetMachine *TM =
      static_cast<const X86TargetMachine *>(&MF.getTarget());
#ifdef __CET__
  bool isJITwithCET = TM->isJIT();
#else
  bool isJITwithCET = false;
#endif
  if (!isCFProtectionSupported && !IndirectBranchTracking && !isJITwithCET)
    return false;

  // True if the current MF was changed and false otherwise.
  bool Changed = false;

  TII = SubTarget.getInstrInfo();
  EndbrOpcode = SubTarget.is64Bit() ? X86::ENDBR64 : X86::ENDBR32;

  if (FineIBT) {
    Changed |= applyFineIBT(MF);
    FixICalls(MF);
    FixDCalls(MF);
  } else {
    Changed |= applyCoarseIBT(MF);
  }
  // Fine IBT on Basic-blocks (JMPs) is not supported yet, thus apply the
  // the regular scheme for both fine and coarse-grained.
  for (auto &MBB : MF) {
    // Find all basic blocks that their address was taken (for example
    // in the case of indirect jump) and add ENDBR instruction.
    if (MBB.hasAddressTaken())
      Changed |= addENDBR(MBB, MBB.begin());

    for (MachineBasicBlock::iterator I = MBB.begin(); I != MBB.end(); ++I) {
      if (I->isCall() && IsCallReturnTwice(I->getOperand(0)))
        Changed |= addENDBR(MBB, std::next(I));
    }

    // Exception handle may indirectly jump to catch pad, So we should add
    // ENDBR before catch pad instructions. For SjLj exception model, it will
    // create a new BB(new landingpad) indirectly jump to the old landingpad.
    if (TM->Options.ExceptionModel == ExceptionHandling::SjLj) {
      for (MachineBasicBlock::iterator I = MBB.begin(); I != MBB.end(); ++I) {
        // New Landingpad BB without EHLabel.
        if (MBB.isEHPad()) {
          if (I->isDebugInstr())
            continue;
          Changed |= addENDBR(MBB, I);
          break;
        } else if (I->isEHLabel()) {
          // Old Landingpad BB (is not Landingpad now) with
          // the the old "callee" EHLabel.
          MCSymbol *Sym = I->getOperand(0).getMCSymbol();
          if (!MF.hasCallSiteLandingPad(Sym))
            continue;
          Changed |= addENDBR(MBB, std::next(I));
          break;
        }
      }
    } else if (MBB.isEHPad()){
      for (MachineBasicBlock::iterator I = MBB.begin(); I != MBB.end(); ++I) {
        if (!I->isEHLabel())
          continue;
        Changed |= addENDBR(MBB, std::next(I));
        break;
      }
    }
  }

  return Changed;
}
