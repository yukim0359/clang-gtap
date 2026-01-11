// This file is made by maeda under gpu-task-parallelism project based on llvm-project.
// Referencing to clang/lib/AST/StmtOpenMP.cpp

#include "clang/AST/StmtGTaP.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Stmt.h"
#include "llvm/ADT/ArrayRef.h"

using namespace clang;

//===----------------------------------------------------------------------===//
// GTaPTaskDirective
//===----------------------------------------------------------------------===//

GTaPTaskDirective *GTaPTaskDirective::Create(const ASTContext &C,
                                             SourceLocation StartLoc,
                                             SourceLocation EndLoc,
                                             Stmt *AssociatedStmt,
                                             Expr *QueueExpr /*nullable*/) {
  void *Mem = C.Allocate(sizeof(GTaPTaskDirective), alignof(GTaPTaskDirective));
  auto *Inst = new (Mem) GTaPTaskDirective(StartLoc, EndLoc);
  Inst->setAssociatedStmt(AssociatedStmt);
  Inst->setQueueExpr(QueueExpr);
  return Inst;
}

GTaPTaskDirective *GTaPTaskDirective::CreateEmpty(const ASTContext &C,
                                                  EmptyShell) {
  void *Mem = C.Allocate(sizeof(GTaPTaskDirective), alignof(GTaPTaskDirective));
  return new (Mem) GTaPTaskDirective();
}

//===----------------------------------------------------------------------===//
// GTaPTaskwaitDirective
//===----------------------------------------------------------------------===//

GTaPTaskwaitDirective *GTaPTaskwaitDirective::Create(const ASTContext &C,
                                                     SourceLocation StartLoc,
                                                     SourceLocation EndLoc,
                                                     Expr *QueueExpr /*nullable*/) {
  void *Mem =
      C.Allocate(sizeof(GTaPTaskwaitDirective), alignof(GTaPTaskwaitDirective));
  auto *Inst = new (Mem) GTaPTaskwaitDirective(StartLoc, EndLoc);
  Inst->setQueueExpr(QueueExpr);
  return Inst;
}

GTaPTaskwaitDirective *GTaPTaskwaitDirective::CreateEmpty(const ASTContext &C,
                                                         EmptyShell) {
  void *Mem =
      C.Allocate(sizeof(GTaPTaskwaitDirective), alignof(GTaPTaskwaitDirective));
  return new (Mem) GTaPTaskwaitDirective();
}

//===----------------------------------------------------------------------===//
// GTaPInitDirective
//===----------------------------------------------------------------------===//

GTaPInitDirective *GTaPInitDirective::Create(const ASTContext &C,
                                             SourceLocation StartLoc,
                                             SourceLocation EndLoc,
                                             StringRef RT, StringRef FN) {
  // Allocate and copy the strings to the AST context for persistence
  // Use IdentifierTable to get or create identifiers, which are interned in the AST context
  StringRef RuntimeType = RT.empty() ? StringRef() : C.Idents.get(RT).getName();
  StringRef FunctionName = FN.empty() ? StringRef() : C.Idents.get(FN).getName();
  
  void *Mem = C.Allocate(sizeof(GTaPInitDirective), alignof(GTaPInitDirective));
  return new (Mem) GTaPInitDirective(StartLoc, EndLoc, RuntimeType, FunctionName);
}

GTaPInitDirective *GTaPInitDirective::CreateEmpty(const ASTContext &C,
                                                  EmptyShell) {
  void *Mem = C.Allocate(sizeof(GTaPInitDirective), alignof(GTaPInitDirective));
  return new (Mem) GTaPInitDirective();
}

//===----------------------------------------------------------------------===//
// GTaPEntryDirective
//===----------------------------------------------------------------------===//

GTaPEntryDirective *GTaPEntryDirective::Create(const ASTContext &C,
                                               SourceLocation StartLoc,
                                               SourceLocation EndLoc,
                                               Stmt *AssociatedStmt) {
  void *Mem = C.Allocate(sizeof(GTaPEntryDirective), alignof(GTaPEntryDirective));
  return new (Mem) GTaPEntryDirective(StartLoc, EndLoc, AssociatedStmt);
}

GTaPEntryDirective *GTaPEntryDirective::CreateEmpty(const ASTContext &C,
                                                    EmptyShell) {
  void *Mem = C.Allocate(sizeof(GTaPEntryDirective), alignof(GTaPEntryDirective));
  return new (Mem) GTaPEntryDirective();
}
