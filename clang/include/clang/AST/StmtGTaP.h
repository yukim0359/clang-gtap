// This file is made by maeda under gpu-task-parallelism project based on llvm-project.
// Referencing to clang/include/clang/AST/StmtOpenMP.h

#ifndef LLVM_CLANG_AST_STMTGTAP_H
#define LLVM_CLANG_AST_STMTGTAP_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/GTaPKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"

namespace clang {

//===----------------------------------------------------------------------===//
// AST classes for directives.
//===----------------------------------------------------------------------===//

/// Base class for all GTaP executable directives.
///
/// This represents directives like:
///   #pragma gtap task
///   #pragma gtap taskwait
///   #pragma gtap init
///   #pragma gtap entry

class GTaPExecutableDirective : public Stmt {
  friend class ASTStmtReader;
  friend class ASTStmtWriter;

  /// Kind of the directive.
  GTaPDirectiveKind Kind = GTaPDirectiveKind::GTaPD_unknown;
  /// Starting location of the directive (directive keyword).
  SourceLocation StartLoc;
  /// Ending location of the directive.
  SourceLocation EndLoc;

protected:
  /// Build instance of directive of class \a SC.
  ///
  /// \param SC Statement class.
  /// \param K Kind of GTaP directive.
  /// \param StartLoc Starting location of the directive (directive keyword).
  /// \param EndLoc Ending location of the directive.
  ///
  GTaPExecutableDirective(StmtClass SC, GTaPDirectiveKind K,
                         SourceLocation StartLoc, SourceLocation EndLoc)
      : Stmt(SC), Kind(K), StartLoc(std::move(StartLoc)),
        EndLoc(std::move(EndLoc)) {}

public:
  /// Returns starting location of directive kind.
  SourceLocation getBeginLoc() const { return StartLoc; }
  /// Returns ending location of directive.
  SourceLocation getEndLoc() const { return EndLoc; }

  /// Set starting location of directive kind.
  ///
  /// \param Loc New starting location of directive.
  ///
  void setLocStart(SourceLocation Loc) { StartLoc = Loc; }
  /// Set ending location of directive.
  ///
  /// \param Loc New ending location of directive.
  ///
  void setLocEnd(SourceLocation Loc) { EndLoc = Loc; }

  GTaPDirectiveKind getDirectiveKind() const { return Kind; }

  static bool classof(const Stmt *S) {
    return S->getStmtClass() >= firstGTaPExecutableDirectiveConstant &&
           S->getStmtClass() <= lastGTaPExecutableDirectiveConstant;
  }

  child_range children() {
    return child_range(child_iterator(), child_iterator());
  }

  const_child_range children() const {
    return const_cast<GTaPExecutableDirective *>(this)->children();
  }
};

/// This represents '#pragma gtap task' directive.
///
/// \code
/// #pragma gtap task queue(0)
///   a = fib(n - 1);
/// \endcode
///
class GTaPTaskDirective : public GTaPExecutableDirective {
  friend class ASTStmtReader;
  friend class GTaPExecutableDirective;

  /// Statement associated with the task (the task body).
  // Stmt *AssociatedStmt = nullptr;
  Stmt *SubStmts[2] = {nullptr, nullptr}; // [0]=AssociatedStmt, [1]=QueueExpr

  /// Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  ///
  GTaPTaskDirective(SourceLocation StartLoc, SourceLocation EndLoc)
      : GTaPExecutableDirective(GTaPTaskDirectiveClass,
                               GTaPDirectiveKind::GTaPD_task,
                               StartLoc, EndLoc) {}

  /// Build an empty directive.
  ///
  explicit GTaPTaskDirective()
      : GTaPExecutableDirective(GTaPTaskDirectiveClass,
                               GTaPDirectiveKind::GTaPD_task,
                               SourceLocation(), SourceLocation()) {}

public:
  /// Creates directive with an associated statement.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param AssociatedStmt Statement, associated with the directive.
  /// \param QueueExpr Queue expression (nullable).
  ///
  static GTaPTaskDirective *Create(const ASTContext &C, SourceLocation StartLoc,
                                   SourceLocation EndLoc, Stmt *AssociatedStmt, Expr *QueueExpr /*nullable*/);

  /// Creates an empty directive.
  ///
  /// \param C AST context.
  ///
  static GTaPTaskDirective *CreateEmpty(const ASTContext &C, EmptyShell);

  /// Returns true if directive has associated statement.
  bool hasAssociatedStmt() const { return SubStmts[0] != nullptr; }

  /// Returns statement associated with the directive.
  const Stmt *getAssociatedStmt() const {
    return SubStmts[0];
  }
  Stmt *getAssociatedStmt() {
    assert(hasAssociatedStmt() &&
           "Expected directive with the associated statement.");
    return SubStmts[0];
  }

  void setAssociatedStmt(Stmt *S) { SubStmts[0] = S; }

  Expr *getQueueExpr() const { return cast_or_null<Expr>(SubStmts[1]); }
  void setQueueExpr(Expr *E) { SubStmts[1] = E; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == GTaPTaskDirectiveClass;
  }

  child_range children() {
    return child_range(child_iterator(SubStmts), child_iterator(SubStmts + 2));
  }
  
  // child_range children() {
  //   if (!AssociatedStmt)
  //     return child_range(child_iterator(), child_iterator());
  //   return child_range(child_iterator(&AssociatedStmt),
  //                      child_iterator(&AssociatedStmt + 1));
  // }

  const_child_range children() const {
    return const_cast<GTaPTaskDirective *>(this)->children();
  }
};

/// This represents '#pragma gtap taskwait' directive.
///
/// \code
/// #pragma gtap taskwait queue(2)
/// \endcode
///
class GTaPTaskwaitDirective : public GTaPExecutableDirective {
  friend class ASTStmtReader;
  friend class GTaPExecutableDirective;

  Expr *QueueExpr = nullptr; // nullable
  unsigned WaitId = ~0u;

  /// Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  ///
  GTaPTaskwaitDirective(SourceLocation StartLoc, SourceLocation EndLoc)
      : GTaPExecutableDirective(GTaPTaskwaitDirectiveClass,
                               GTaPDirectiveKind::GTaPD_taskwait,
                               StartLoc, EndLoc) {}

  /// Build an empty directive.
  ///
  explicit GTaPTaskwaitDirective()
      : GTaPExecutableDirective(GTaPTaskwaitDirectiveClass,
                               GTaPDirectiveKind::GTaPD_taskwait,
                               SourceLocation(), SourceLocation()) {}

public:
  /// Creates directive.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param QueueExpr Queue expression (nullable).

  static GTaPTaskwaitDirective * Create(
    const ASTContext &C, SourceLocation StartLoc, 
    SourceLocation EndLoc, Expr *QueueExpr /*nullable*/
  );

  /// Creates an empty directive.
  ///
  /// \param C AST context.
  ///
  static GTaPTaskwaitDirective *CreateEmpty(const ASTContext &C, EmptyShell);

  Expr *getQueueExpr() const { return cast_or_null<Expr>(QueueExpr); }
  void setQueueExpr(Expr *E) { QueueExpr = E; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == GTaPTaskwaitDirectiveClass;
  }

  void setWaitId(unsigned I) { WaitId = I; }
  unsigned getWaitId() const { return WaitId; }
};

/// This represents '#pragma gtap init' directive.
///
/// \code
/// #pragma gtap init
/// \endcode
///
class GTaPInitDirective : public GTaPExecutableDirective {
  friend class ASTStmtReader;
  friend class GTaPExecutableDirective;

  /// Runtime type identifier (e.g., "thread", "block").
  StringRef RuntimeType;
  /// Function name identifier (e.g., "fib").
  StringRef FunctionName;

  /// Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param RT Runtime type identifier.
  /// \param FN Function name identifier.
  ///
  GTaPInitDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                   StringRef RT, StringRef FN)
      : GTaPExecutableDirective(GTaPInitDirectiveClass,
                               GTaPDirectiveKind::GTaPD_init,
                               StartLoc,
                               EndLoc),
        RuntimeType(RT), FunctionName(FN) {}

  /// Build an empty directive.
  ///
  explicit GTaPInitDirective()
      : GTaPExecutableDirective(GTaPInitDirectiveClass,
                               GTaPDirectiveKind::GTaPD_init,
                               SourceLocation(),
                               SourceLocation()) {}

public:
  /// Creates directive.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param RT Runtime type identifier.
  /// \param FN Function name identifier.
  ///
  static GTaPInitDirective *Create(const ASTContext &C, SourceLocation StartLoc, SourceLocation EndLoc,
                                  StringRef RT, StringRef FN);

  /// Creates an empty directive.
  ///
  /// \param C AST context.
  ///
  static GTaPInitDirective *CreateEmpty(const ASTContext &C, EmptyShell);

  /// Get the runtime type identifier.
  StringRef getRuntimeType() const { return RuntimeType; }

  /// Get the function name identifier.
  StringRef getFunctionName() const { return FunctionName; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == GTaPInitDirectiveClass;
  }
};

/// This represents '#pragma gtap entry' directive.
///
/// \code
/// #pragma gtap entry
/// result = fib(n);
/// \endcode
///
class GTaPEntryDirective : public GTaPExecutableDirective {
  friend class ASTStmtReader;
  friend class GTaPExecutableDirective;

  /// Statement associated with the entry (the expression after the pragma).
  Stmt *AssociatedStmt = nullptr;

  /// Build directive with the given start and end location.
  ///
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending location of the directive.
  /// \param AssociatedStmt Statement associated with the directive.
  ///
  GTaPEntryDirective(SourceLocation StartLoc, SourceLocation EndLoc,
                    Stmt *AssociatedStmt)
      : GTaPExecutableDirective(GTaPEntryDirectiveClass,
                               GTaPDirectiveKind::GTaPD_entry,
                               StartLoc,
                               EndLoc),
        AssociatedStmt(AssociatedStmt) {}

  /// Build an empty directive.
  ///
  explicit GTaPEntryDirective()
      : GTaPExecutableDirective(GTaPEntryDirectiveClass,
                               GTaPDirectiveKind::GTaPD_entry,
                               SourceLocation(),
                               SourceLocation()) {}

public:
  /// Creates directive with an associated statement.
  ///
  /// \param C AST context.
  /// \param StartLoc Starting location of the directive kind.
  /// \param EndLoc Ending Location of the directive.
  /// \param AssociatedStmt Statement, associated with the directive.
  ///
  static GTaPEntryDirective *Create(const ASTContext &C, SourceLocation StartLoc, 
                                   SourceLocation EndLoc, Stmt *AssociatedStmt);

  /// Creates an empty directive.
  ///
  /// \param C AST context.
  ///
  static GTaPEntryDirective *CreateEmpty(const ASTContext &C, EmptyShell);

  /// Returns true if directive has associated statement.
  bool hasAssociatedStmt() const { return AssociatedStmt != nullptr; }

  /// Returns statement associated with the directive.
  const Stmt *getAssociatedStmt() const {
    return const_cast<GTaPEntryDirective *>(this)->getAssociatedStmt();
  }
  Stmt *getAssociatedStmt() {
    assert(hasAssociatedStmt() &&
           "Expected directive with the associated statement.");
    return AssociatedStmt;
  }

  void setAssociatedStmt(Stmt *S) { AssociatedStmt = S; }

  static bool classof(const Stmt *T) {
    return T->getStmtClass() == GTaPEntryDirectiveClass;
  }

  child_range children() {
    if (!AssociatedStmt)
      return child_range(child_iterator(), child_iterator());
    return child_range(child_iterator(&AssociatedStmt),
                       child_iterator(&AssociatedStmt + 1));
  }

  const_child_range children() const {
    return const_cast<GTaPEntryDirective *>(this)->children();
  }
};

}

#endif
