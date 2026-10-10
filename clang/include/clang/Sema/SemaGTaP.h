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
#include <utility>
#include <vector>

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
  // Called after parsing a function declaration to consume a pending pragma.
  void ActOnFunctionDeclaration(FunctionDecl *FD);

  // Called when ActOnStartOfFunctionDef to attach pending GTaP function attribute
  void ActOnStartOfFunctionDef(FunctionDecl *FD);

  /// Return true if any declaration in the function's redeclaration chain is
  /// marked as a GTaP task function.
  bool isGTaPTaskFunction(const FunctionDecl *FD) const {
    if (!FD)
      return false;
    for (const FunctionDecl *Redecl : FD->redecls())
      if (Redecl->hasAttr<GTaPFunctionAttr>())
        return true;
    return false;
  }

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

  /// Check whether the associated statement of a GTaP task directive is
  /// currently being parsed.  Direct calls to task functions are valid only
  /// in this context (or in an entry directive).
  bool isInGTaPTaskDirective() const { return InGTaPTaskDirective; }

  void pushGTaPTaskDirective() {
    assert(!InGTaPTaskDirective && "nested GTaP task directives are not allowed");
    InGTaPTaskDirective = true;
  }

  void popGTaPTaskDirective() {
    assert(InGTaPTaskDirective && "GTaP task directive flag underflow");
    InGTaPTaskDirective = false;
  }

  /// Transform a user-authored GTaP task function into its state-machine-driven
  /// representation at the AST level.
  StmtResult TransformTaskFunctionBody(FunctionDecl *FD, CompoundStmt *Body);

  /// Get cached task info for a function (for use in TransformGTaPTaskDirective)
  GTaPTaskFunctionInfo &getCachedTaskInfo(FunctionDecl *FD);

  /// Get or create the entry function for a user-authored GTaP task function
  FunctionDecl *getOrCreateStateMachineFunction(FunctionDecl *UserFD,
                                                QualType VoidTy,
                                                QualType VoidPtrTy,
                                                QualType IntTy,
                                                QualType TaskCtxPtrTy);

  /// Record a generated task-data layout in this translation unit.
  void noteTaskRecordSize(uint64_t Bytes, uint64_t Align);
  void noteBlockTaskRecordLayout(uint64_t FixedBytes,
                                 uint64_t LaneStorageBytes,
                                 uint64_t Align);
  void noteEntryResultSize(uint64_t Bytes);

  /// Notify the AST consumer about compiler-generated metadata definitions
  /// after every task-data layout in the translation unit is known.
  void ActOnEndOfTranslationUnit();

private:
  // Get the AST context.
  ASTContext &getASTContext();
  
  /// Flag indicating if we are currently inside a GTaP entry directive.
  bool InGTaPEntryDirective = false;

  /// Flag indicating that a '#pragma gtap task' associated statement is being
  /// parsed.
  bool InGTaPTaskDirective = false;

  /// Cache of analysed task functions, keyed by the original declaration.
  llvm::DenseMap<const FunctionDecl *, GTaPTaskFunctionInfo> CachedTaskInfos;

  uint64_t AutoTaskDataSize = 0;
  uint64_t AutoTaskDataAlign = 1;
  VarDecl *AutoTaskDataSizeDecl = nullptr;
  VarDecl *AutoTaskDataAlignDecl = nullptr;
  std::vector<std::pair<uint64_t, uint64_t>> AutoBlockTaskDataLayouts;
  VarDecl *AutoBlockTaskDataSizesDecl = nullptr;
  uint64_t AutoEntryResultSize = 1;
  VarDecl *AutoEntryResultSizeDecl = nullptr;
};

}

#endif
