// This file is made by maeda under gpu-task-parallelism project based on llvm-project.
// Referencing to clang/include/clang/Sema/SemaOpenMP.h

#ifndef LLVM_CLANG_SEMA_SEMAGTAP_H
#define LLVM_CLANG_SEMA_SEMAGTAP_H

#include "clang/AST/ASTFwd.h"
#include "clang/AST/GTaPTaskInfo.h"
#include "clang/AST/StmtGTaP.h"
#include "clang/Basic/GTaPKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/SemaBase.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include <cstdint>

namespace clang {
class ASTContext;
class GTaPTaskFunctionAnalyzer;
}

namespace clang {

class DeclContext;
class Scope;

class SemaGTaP : public SemaBase {
public:
  SemaGTaP(Sema &S);

  friend class Parser;
  friend class Sema;

  /// Act on a GTaP executable directive.
  ///
  /// \param DKind The directive kind.
  /// \param AStmt The associated statement (for task directive).
  /// \param StartLoc The start location.
  /// \param EndLoc The end location.
  ///
  /// \returns Statement for finished GTaP region.
  StmtResult ActOnGTaPExecutableDirective(GTaPDirectiveKind DKind,
                                          Stmt *AStmt,
                                          SourceLocation StartLoc,
                                          SourceLocation EndLoc,
                                          Expr *QueueExpr);

  /// Called on well-formed '#pragma gtap task' after parsing
  /// of the associated statement.
  StmtResult ActOnGTaPTaskDirective(Stmt *AStmt,
                                    SourceLocation StartLoc,
                                    SourceLocation EndLoc,
                                    Expr *QueueExpr);

  /// Called on well-formed '#pragma gtap taskwait'.
  StmtResult ActOnGTaPTaskwaitDirective(SourceLocation StartLoc,
                                        SourceLocation EndLoc,
                                        Expr *QueueExpr);

  /// Called on well-formed '#pragma gtap init'.
  ///
  /// \param StartLoc Starting location of the directive.
  /// \param EndLoc Ending location of the directive.
  /// \param RT Runtime type identifier (e.g., "thread", "block").
  /// \param FN Function name identifier (e.g., "fib").
  ///
  StmtResult ActOnGTaPInitDirective(SourceLocation StartLoc,
                                    SourceLocation EndLoc,
                                    StringRef RT,
                                    StringRef FN);

  /// Called on well-formed '#pragma gtap entry'.
  ///
  /// \param StartLoc Starting location of the directive.
  /// \param EndLoc Ending location of the directive.
  /// \param AStmt The associated statement (expression after entry pragma).
  ///
  StmtResult ActOnGTaPEntryDirective(SourceLocation StartLoc,
                                     SourceLocation EndLoc,
                                     Stmt *AStmt);

  // Called on well-formed '#pragma gtap function'.
  // Called when ActOnStartOfFunctionDef to attach pending GTaP function attribute
  void ActOnStartOfFunctionDef(FunctionDecl *FD);

  // Check if there is a pending GTaP function pragma
  bool hasPendingGTaPFunctionPragma() const {
    return PendingFunctionPragmaLoc.isValid();
  }

  // Clear the pending GTaP function pragma
  void clearPendingGTaPFunctionPragma() {
    PendingFunctionPragmaLoc = SourceLocation();
  }

  // Pending function pragma: set by #pragma gtap function at file scope
  // Made public so PragmaHandler can set it directly
  SourceLocation PendingFunctionPragmaLoc;

  /// Check if we are currently inside a GTaP entry directive.
  bool isInGTaPEntryDirective() const {
    return InGTaPEntryDirective;
  }

  /// Set flag when entering GTaP entry directive.
  void pushGTaPEntryDirective() {
    assert(!InGTaPEntryDirective && "nested GTaP entry directives are not allowed");
    InGTaPEntryDirective = true;
  }

  /// Clear flag when exiting GTaP entry directive.
  void popGTaPEntryDirective() {
    assert(InGTaPEntryDirective && "GTaP entry directive flag underflow");
    InGTaPEntryDirective = false;
  }

  /// Transform a user-authored GTaP task function into its state-machine-driven
  /// representation at the AST level.
  StmtResult TransformTaskFunctionBody(FunctionDecl *FD, CompoundStmt *Body);

  /// Get cached task info for a function (for use in TransformGTaPTaskDirective)
  GTaPTaskFunctionInfo &getCachedTaskInfo(FunctionDecl *FD) {
    // Create if it doesn't exist
    if (CachedTaskInfos.find(FD) == CachedTaskInfos.end()) {
      ASTContext &Ctx = getASTContext();
      GTaPTaskFunctionAnalyzer Analyzer(Ctx, FD);
      CachedTaskInfos[FD] = Analyzer.analyze();
    }
    return CachedTaskInfos[FD];
  }

  /// Get or create the entry function for a user-authored GTaP task function
  FunctionDecl *getOrCreateStateMachineFunction(FunctionDecl *UserFD,
                                                QualType VoidTy,
                                                QualType VoidPtrTy,
                                                QualType IntTy,
                                                QualType TaskCtxPtrTy);

  /// Record the largest generated task-data record size in this translation unit.
  void noteTaskRecordSize(uint64_t Bytes);

private:
  // Get the AST context.
  ASTContext &getASTContext();
  
  /// Flag indicating if we are currently inside a GTaP entry directive.
  bool InGTaPEntryDirective = false;

  /// Cache of analysed task functions, keyed by the original declaration.
  llvm::DenseMap<const FunctionDecl *, GTaPTaskFunctionInfo> CachedTaskInfos;

  uint64_t AutoTaskDataSize = 1;
  VarDecl *AutoTaskDataSizeDecl = nullptr;
};

}

#endif
