// This file is made by maeda under gpu-task-parallelism project based on llvm-project.

#ifndef LLVM_CLANG_AST_GTAPTASKINFO_H
#define LLVM_CLANG_AST_GTAPTASKINFO_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtGTaP.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/SaveAndRestore.h"
#include "clang/Analysis/CFG.h"
#include <vector>

namespace clang {

/// Metadata describing a single GTaP directive encountered while analysing a task function.
struct GTaPDirectiveInfo {
  SourceLocation Location;
  enum Kind { Task, Taskwait } DirectiveKind = Task;
  const Stmt *AssociatedStmt = nullptr;
};

/// Aggregated information about a task function that is required for lowering.
struct GTaPTaskFunctionInfo {
  FunctionDecl *Func = nullptr;
  std::vector<GTaPDirectiveInfo> Directives;
  std::vector<ParmVarDecl *> Parameters;
  std::vector<VarDecl *> CapturedVariables;
  QualType ReturnType;
  RecordDecl *TaskRecord = nullptr;
  bool TaskRecordInvalid = false;
  std::vector<FieldDecl *> ParameterFields;
  std::vector<FieldDecl *> CapturedFields;
  FieldDecl *StateField = nullptr;
  FieldDecl *ResultField = nullptr;
  FieldDecl *ResultDstField = nullptr;
  FieldDecl *SpawningThreadField = nullptr;
  llvm::DenseSet<const VarDecl*> SpillSet;
  llvm::DenseSet<const VarDecl*> HoistOnlySet;
  FunctionDecl *StateMachineFD = nullptr;
};

struct UseDef {
  llvm::SmallPtrSet<const VarDecl*, 16> Uses;
  llvm::SmallPtrSet<const VarDecl*, 16> Defs;
};

class UseDefVisitor : public RecursiveASTVisitor<UseDefVisitor> {
public:
  UseDefVisitor(ASTContext &Ctx, UseDef &UD) : Ctx(Ctx), UD(UD) {}

  bool VisitDeclRefExpr(DeclRefExpr *DRE) {
    if (InWriteContext) return true;
    if (auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
      if (VD->hasLocalStorage() && VD->isLocalVarDecl())
        UD.Uses.insert(VD);
    }
    return true;
  }

  bool TraverseBinaryOperator(BinaryOperator *BO) {
    if (!BO) return true;

    if (BO->isAssignmentOp()) {
      TraverseStmt(BO->getRHS());

      if (BO->isCompoundAssignmentOp()) {
        if (auto *DRE = dyn_cast<DeclRefExpr>(BO->getLHS()->IgnoreParenImpCasts())) {
          if (auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
            if (VD->hasLocalStorage() && VD->isLocalVarDecl())
              UD.Uses.insert(VD);
          }
        }
      }

      llvm::SaveAndRestore<bool> SR(InWriteContext, true);
      TraverseStmt(BO->getLHS());

      if (auto *DRE = dyn_cast<DeclRefExpr>(BO->getLHS()->IgnoreParenImpCasts())) {
        if (auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
          if (VD->hasLocalStorage() && VD->isLocalVarDecl())
            UD.Defs.insert(VD);
        }
      }
      return true;
    }

    return RecursiveASTVisitor::TraverseBinaryOperator(BO);
  }

  bool TraverseUnaryOperator(UnaryOperator *UO) {
    if (!UO) return true;
    if (UO->isIncrementDecrementOp()) {
      TraverseStmt(UO->getSubExpr());
      if (auto *DRE = dyn_cast<DeclRefExpr>(UO->getSubExpr()->IgnoreParenImpCasts())) {
        if (auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
          if (VD->hasLocalStorage() && VD->isLocalVarDecl()) {
            UD.Uses.insert(VD);
            UD.Defs.insert(VD);
          }
        }
      }
      return true;
    }
    return RecursiveASTVisitor::TraverseUnaryOperator(UO);
  }

private:
  [[maybe_unused]] ASTContext &Ctx;
  UseDef &UD;
  bool InWriteContext = false;
};

/// AST visitor that gathers \ref GTaPTaskFunctionInfo for a given function.
class GTaPTaskFunctionAnalyzer
    : public RecursiveASTVisitor<GTaPTaskFunctionAnalyzer> {
public:
  GTaPTaskFunctionAnalyzer(ASTContext &Ctx, FunctionDecl *TargetFunc)
      : Ctx(Ctx), TargetFunc(TargetFunc) {}

  bool VisitGTaPTaskDirective(GTaPTaskDirective *D) {
    GTaPDirectiveInfo Info;
    Info.Location = D->getBeginLoc();
    Info.DirectiveKind = GTaPDirectiveInfo::Task;
    Info.AssociatedStmt = D->getAssociatedStmt();
    Result.Directives.push_back(Info);
    return true;
  }

  bool VisitGTaPTaskwaitDirective(GTaPTaskwaitDirective *D) {
    GTaPDirectiveInfo Info;
    Info.Location = D->getBeginLoc();
    Info.DirectiveKind = GTaPDirectiveInfo::Taskwait;
    Info.AssociatedStmt = nullptr;
    Result.Directives.push_back(Info);
    return true;
  }

  bool VisitVarDecl(VarDecl *VD) {
    if (VD && VD->isLocalVarDecl())
      LocalVariables.push_back(VD);
    return true;
  }

  GTaPTaskFunctionInfo analyze(Stmt *BodyOverride = nullptr) {
    Result.Func = TargetFunc;
    Result.ReturnType = TargetFunc->getReturnType();
    for (auto *Param : TargetFunc->parameters())
      Result.Parameters.push_back(Param);
  
    if (BodyOverride) {
      TraverseStmt(BodyOverride);
    } else if (TargetFunc->hasBody()) {
      TraverseStmt(TargetFunc->getBody());
    } else if (FunctionDecl *Def = TargetFunc->getDefinition()) {
      if (Def->hasBody()) TraverseStmt(Def->getBody());
    }
  
    determineCapturedVariables(BodyOverride);
    return Result;
  }

private:
  void determineCapturedVariables(Stmt *BodyOverride) {
    Result.CapturedVariables.clear();

    if (!TargetFunc) return;
  
    // 1) Specify the target body to analyze
    Stmt *Body = BodyOverride;
    if (!Body) {
      if (TargetFunc->hasBody()) Body = TargetFunc->getBody();
      else if (FunctionDecl *Def = TargetFunc->getDefinition())
        if (Def->hasBody()) Body = Def->getBody();
    }
    if (!Body) {
      // llvm::errs() << "[GTaP] No body available for " << TargetFunc->getName() << "\n";
      return;
    }
  
    // 2) Determine if there is a taskwait in both "Directives" and "Body scanning"
    bool HasTaskwait = false;
    for (auto &DI : Result.Directives) {
      if (DI.DirectiveKind == GTaPDirectiveInfo::Taskwait) { HasTaskwait = true; break; }
    }
    if (!HasTaskwait) {
      struct Finder : RecursiveASTVisitor<Finder> {
        bool Found = false;
        bool VisitGTaPTaskwaitDirective(GTaPTaskwaitDirective *) { Found = true; return false; }
      } F;
      F.TraverseStmt(Body);
      HasTaskwait = F.Found;
    }
  
    if (!HasTaskwait) {
      Result.CapturedVariables.clear();
      Result.SpillSet.clear();
      Result.HoistOnlySet.clear();
      return;
    }
  
    // 3) Build CFG with Body (override)
    CFG::BuildOptions BO;
    std::unique_ptr<CFG> Cfg = CFG::buildCFG(TargetFunc, Body, &Ctx, BO);
    if (!Cfg) {
      // llvm::errs() << "[GTaP] CFG construction failed for function: "
      //              << TargetFunc->getName() << "\n";
      // Fallback (if there is a taskwait but CFG is not built, capture all local variables safely)
      llvm::SmallPtrSet<const VarDecl *, 16> Seen;
      for (auto *VD : LocalVariables) {
        if (!VD || Seen.contains(VD)) continue;
        Result.CapturedVariables.push_back(VD);
        Seen.insert(VD);
      }
      return;
    }

    // llvm::errs() << "[GTaP] CFG construction succeeded for function: "
    //               << TargetFunc->getName() << "\n";
    // for (const CFGBlock *B : *Cfg) {
    //   llvm::errs() << "Block B" << B->getBlockID() << " size=" << B->size() << "\n";
    //   if (const Stmt *T = B->getTerminatorStmt()) {
    //     llvm::errs() << "  Terminator=" << T->getStmtClassName() << "\n";
    //   }
    //   for (const auto &E : *B)
    //     llvm::errs() << "  ElemKind=" << (int)E.getKind() << "\n";
    //   for (const auto &E : *B) {
    //     if (auto CS = E.getAs<CFGStmt>()) {
    //       const Stmt *S = CS->getStmt();
    //       llvm::errs() << "  CFGStmt: " << S->getStmtClassName() << "\n";
    //     } else {
    //       llvm::errs() << "  ElemKind=" << (int)E.getKind() << "\n";
    //     }
    //   }
    // }

    using VarSet = llvm::SmallPtrSet<const VarDecl *, 32>;

    auto setEqual = [](const VarSet &A, const VarSet &B) -> bool {
      if (A.size() != B.size()) return false;
      for (auto *V : A) if (!B.contains(V)) return false;
      return true;
    };

    auto unionInto = [](VarSet &Dst, const VarSet &Src) {
      for (auto *V : Src) Dst.insert(V);
    };

    llvm::SmallPtrSet<const Stmt*, 32> TaskAssocStmts;

    // First, collect Directives (put AssociatedStmt)
    for (auto &DI : Result.Directives) {
      if (DI.DirectiveKind == GTaPDirectiveInfo::Task && DI.AssociatedStmt)
        TaskAssocStmts.insert(DI.AssociatedStmt);
    }

    auto isGTaPTaskSpawnAssign = [&](const Stmt *S, UseDef &UD) -> void {
      if (!S) return;
      if (!TaskAssocStmts.contains(S)) return;
    
      auto *BO = dyn_cast<BinaryOperator>(S);
      if (!BO || BO->getOpcode() != BO_Assign) return;
    
      auto *Call = dyn_cast<CallExpr>(BO->getRHS()->IgnoreParenImpCasts());
      if (!Call) return;
    
      const FunctionDecl *Callee = Call->getDirectCallee();
      if (!Callee || !Callee->hasAttr<GTaPFunctionAttr>()) return;
    
      // Remove the Var from Defs because it will be set after the taskwait
      if (auto *DRE = dyn_cast<DeclRefExpr>(BO->getLHS()->IgnoreParenImpCasts())) {
        if (auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
          UD.Defs.erase(VD);
          // UD.Uses.erase(VD);  // Is this necessary?
        }
      }
    };

    auto computeStmtUseDef = [&](const Stmt *S) -> UseDef {
      UseDef UD;
      if (!S) return UD;

      UseDefVisitor V(Ctx, UD);
      V.TraverseStmt(const_cast<Stmt *>(S));

      if (auto *DS = dyn_cast<DeclStmt>(S)) {
        for (Decl *D : DS->decls())
          if (auto *VD = dyn_cast<VarDecl>(D))
            if (VD->hasLocalStorage() && VD->isLocalVarDecl())
              UD.Defs.insert(VD);
      }

      isGTaPTaskSpawnAssign(S, UD);
      return UD;
    };

    struct BlockUD { VarSet Use, Def; };
    llvm::DenseMap<const CFGBlock *, BlockUD> BlockUseDef;
    llvm::DenseMap<const CFGBlock *, VarSet> BlockMentionGen; // Uses \cup Defs

    // (A) Calculate Use/Def for each block (Use is "used before Def in the block")
    for (const CFGBlock *B : *Cfg) {
      if (!B) continue;
      BlockUD BUD;
      VarSet MG;
      for (const auto &Elem : *B) {
        if (auto CS = Elem.getAs<CFGStmt>()) {
          const Stmt *S = CS->getStmt();
          UseDef UD = computeStmtUseDef(S);

          // Use += (UD.Uses - BUD.Def)
          for (auto *U : UD.Uses)
            if (!BUD.Def.contains(U)) BUD.Use.insert(U);
          // Def += UD.Defs
          for (auto *D : UD.Defs)
            BUD.Def.insert(D);

          // MG += (UD.Uses \cup UD.Defs)
          for (auto *U : UD.Uses) MG.insert(U);
          for (auto *D : UD.Defs) MG.insert(D);
        }
      }
      BlockUseDef[B] = std::move(BUD);
      BlockMentionGen[B] = std::move(MG);
    }

    // (B) Liveness: LiveIn/LiveOut by fixed point calculation
    llvm::DenseMap<const CFGBlock *, VarSet> LiveIn, LiveOut;
    bool Changed = true;
    while (Changed) {
      Changed = false;
      for (const CFGBlock *B : *Cfg) {
        if (!B) continue;

        VarSet NewOut;
        for (auto SI = B->succ_begin(); SI != B->succ_end(); ++SI) {
          if (const CFGBlock *Succ = *SI) {
            unionInto(NewOut, LiveIn[Succ]);
          }
        }

        const BlockUD &BUD = BlockUseDef[B];
        VarSet NewIn = BUD.Use;
        for (auto *V : NewOut)
          if (!BUD.Def.contains(V)) NewIn.insert(V);

        if (!setEqual(NewOut, LiveOut[B]) || !setEqual(NewIn, LiveIn[B])) {
          LiveOut[B] = std::move(NewOut);
          LiveIn[B]  = std::move(NewIn);
          Changed = true;
        }
      }
    }

    // (B) Calculate MentionGen/MentionKill for each block
    llvm::DenseMap<const CFGBlock *, VarSet> MentionIn, MentionOut;
    bool Changed2 = true;
    while (Changed2) {
      Changed2 = false;
      for (const CFGBlock *B : *Cfg) {
        if (!B) continue;
    
        VarSet NewOut;
        for (auto SI = B->succ_begin(); SI != B->succ_end(); ++SI)
          if (const CFGBlock *Succ = *SI)
            unionInto(NewOut, MentionIn[Succ]);
    
        VarSet NewIn = NewOut;
        unionInto(NewIn, BlockMentionGen[B]);
    
        if (!setEqual(NewOut, MentionOut[B]) || !setEqual(NewIn, MentionIn[B])) {
          MentionOut[B] = std::move(NewOut);
          MentionIn[B]  = std::move(NewIn);
          Changed2 = true;
        }
      }
    }

    // (C) Traverse each block from the end, and capture the Live that is "immediately after the taskwait"
    SourceManager &SM = Ctx.getSourceManager();
    auto before = [&](SourceLocation A, SourceLocation B) -> bool {
      if (A.isInvalid() || B.isInvalid()) return false;
      A = SM.getSpellingLoc(A);
      B = SM.getSpellingLoc(B);
      return SM.isBeforeInTranslationUnit(A, B);
    };
    
    llvm::SmallPtrSet<const VarDecl *, 32> SpillSetLocal;
    llvm::SmallPtrSet<const VarDecl *, 32> HoistOnlyLocal;
    
    for (const CFGBlock *B : *Cfg) {
      if (!B) continue;
      VarSet Live = LiveOut[B];
      VarSet Mention = MentionOut[B];
    
      for (auto EI = B->rbegin(); EI != B->rend(); ++EI) {
        if (auto CS = EI->getAs<CFGStmt>()) {
          const Stmt *S = CS->getStmt();
          if (auto *TW = dyn_cast<GTaPTaskwaitDirective>(S)) {
            for (auto *V : Live) SpillSetLocal.insert(V);
    
            SourceLocation TWLoc = TW->getBeginLoc();
            for (auto *V : Mention) {
              SourceLocation DLoc = V->getBeginLoc();
              if (before(DLoc, TWLoc))
                HoistOnlyLocal.insert(V);
            }
          }
    
          UseDef UD = computeStmtUseDef(S);
    
          // Live <- (Live - Def) \cup Use
          for (auto *D : UD.Defs) Live.erase(D);
          for (auto *U : UD.Uses) Live.insert(U);
    
          // Mention <- Mention \cup (Uses \cup Defs)
          for (auto *U : UD.Uses) Mention.insert(U);
          for (auto *D : UD.Defs) Mention.insert(D);
        }
      }
    }

    Result.SpillSet.clear();
    Result.HoistOnlySet.clear();
    Result.CapturedVariables.clear();
    
    for (auto *V : SpillSetLocal) Result.SpillSet.insert(V);
    for (auto *V : HoistOnlyLocal) Result.HoistOnlySet.insert(V);
    
    // Captured = union
    llvm::SmallPtrSet<const VarDecl*, 32> Union;
    for (auto *V : SpillSetLocal) Union.insert(V);
    for (auto *V : HoistOnlyLocal) Union.insert(V);
    for (auto *VD : Union)
      Result.CapturedVariables.push_back(const_cast<VarDecl*>(VD));
  }

  ASTContext &Ctx;
  FunctionDecl *TargetFunc = nullptr;
  GTaPTaskFunctionInfo Result;
  std::vector<VarDecl *> LocalVariables;
};

} // namespace clang

#endif // LLVM_CLANG_AST_GTAPTASKINFO_H
