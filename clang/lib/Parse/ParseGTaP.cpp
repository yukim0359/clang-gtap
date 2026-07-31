// This file is made by maeda under gpu-task-parallelism project based on llvm-project.
// Referencing to clang/lib/Parse/ParseOpenMP.cpp

#include "clang/Basic/DiagnosticParse.h"
#include "clang/Basic/GTaPKinds.h"
#include "clang/Basic/TokenKinds.h"
#include "clang/Parse/Parser.h"
#include "clang/Parse/RAIIObjectsForParser.h"
#include "clang/Sema/SemaGTaP.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

using namespace clang;

// Parse a gtap directive kind from the current token stream.
static GTaPDirectiveKind parseGTaPDirectiveKind(Parser &P) {
  Token Tok = P.getCurToken();
  if (Tok.isAnnotation()) {
    return GTaPDirectiveKind::GTaPD_unknown;
  }
  std::string DirName = P.getPreprocessor().getSpelling(Tok);
  P.ConsumeToken();
  return getGTaPDirectiveKind(DirName);
}

// Parse clause: queue(<expr>)
// Returns parsed Expr on success, nullptr on error.
static Expr *parseClauseWithExpr(Parser &P, StringRef ExpectedClauseName) {
  if (!P.getCurToken().is(tok::identifier))
    return nullptr;

  IdentifierInfo *II = P.getCurToken().getIdentifierInfo();
  StringRef ClauseName = II ? II->getName() : StringRef();
  if (ClauseName != ExpectedClauseName)
    return nullptr;

  P.ConsumeToken(); // 'queue'

  if (!P.getCurToken().is(tok::l_paren)) {
    P.Diag(P.getCurToken().getLocation(), diag::err_expected) << tok::l_paren;
    P.SkipUntil(tok::annot_pragma_gtap_end, Parser::StopBeforeMatch);
    return nullptr;
  }
  P.ConsumeToken(); // '('

  ExprResult ER = P.ParseAssignmentExpression();
  if (ER.isInvalid()) {
    P.SkipUntil(tok::r_paren, Parser::StopBeforeMatch);
    if (P.getCurToken().is(tok::r_paren)) P.ConsumeToken();
    return nullptr;
  }

  if (!P.getCurToken().is(tok::r_paren)) {
    P.Diag(P.getCurToken().getLocation(), diag::err_expected) << tok::r_paren;
    P.SkipUntil(tok::annot_pragma_gtap_end, Parser::StopBeforeMatch);
    return nullptr;
  }
  P.ConsumeToken(); // ')'

  return ER.get();
}

// Parse an executable gtap directive.
StmtResult Parser::ParseGTaPExecutableDirective() {
  // Consume the annotation token.
  assert(getCurToken().is(tok::annot_pragma_gtap) && "Expected annot_pragma_gtap token");
  SourceLocation StartLoc = getCurToken().getLocation();
  ConsumeToken(); // Consume 'annot_pragma_gtap'

  GTaPDirectiveKind DKind = parseGTaPDirectiveKind(*this);
  if (DKind == GTaPDirectiveKind::GTaPD_unknown) {
    Diag(StartLoc, diag::err_gtap_unknown_directive);
    SkipUntil(tok::annot_pragma_gtap_end, StopBeforeMatch);
    return StmtError();
  }

  // Parse arguments for init directive: #pragma gtap init
  StringRef RuntimeType;
  StringRef FunctionName;

  Expr *QueueExpr = nullptr;
  if (DKind == GTaPDirectiveKind::GTaPD_task || DKind == GTaPDirectiveKind::GTaPD_taskwait) {
    while (getCurToken().is(tok::identifier)) {
      StringRef Name = getCurToken().getIdentifierInfo()->getName();
      if (Name == "queue") {
        Expr *E = parseClauseWithExpr(*this, "queue");
        if (!E) return StmtError();
        QueueExpr = E;
        continue;
      }
      break;
    }
  }

  // Consume the end annotation token.
  SourceLocation EndLoc = getCurToken().getLocation();
  if (getCurToken().is(tok::annot_pragma_gtap_end)) {
    ConsumeToken();
  } else {
    SkipUntil(tok::annot_pragma_gtap_end, StopBeforeMatch);
    if (getCurToken().is(tok::annot_pragma_gtap_end)) {
      EndLoc = getCurToken().getLocation();
      ConsumeToken();
    }
  }

  // Parse the associated statement for task or entry directive (after consuming end token).
  // This allows the statement to be on the next line, similar to OpenMP:
  //   #pragma gtap entry
  //   result = fib(n);
  Stmt *AStmt = nullptr;
  if (DKind == GTaPDirectiveKind::GTaPD_task || DKind == GTaPDirectiveKind::GTaPD_entry) {
    if (DKind == GTaPDirectiveKind::GTaPD_task) {
      getActions().GTaP().pushGTaPTaskDirective();
    }

    // For entry directive, increment depth before parsing to allow host->device calls
    if (DKind == GTaPDirectiveKind::GTaPD_entry) {
      getActions().GTaP().pushGTaPEntryDirective();
    }
    
    // For entry directive, suppress diagnostics since the statement may contain
    // errors (e.g., calling a transformed GTaP function with wrong signature)
    // Sema will recover from these errors and generate correct code
    bool SuppressDiags = (DKind == GTaPDirectiveKind::GTaPD_entry);
    
    if (SuppressDiags) {
      // Temporarily suppress error diagnostics
      Diags.setSuppressAllDiagnostics(true);
    }
    
    // Parse the associated statement (task body or entry expression).
    // For now, we'll parse a single statement.
    StmtResult Body = ParseStatement();
    
    if (SuppressDiags) {
      // Restore diagnostics
      Diags.setSuppressAllDiagnostics(false);
    }
    
    // Decrement depth after parsing
    if (DKind == GTaPDirectiveKind::GTaPD_entry) {
      getActions().GTaP().popGTaPEntryDirective();
    }
    if (DKind == GTaPDirectiveKind::GTaPD_task) {
      getActions().GTaP().popGTaPTaskDirective();
    }
    
    // For entry directive, accept the statement even if it has errors
    // Sema will handle error recovery
    if (Body.isInvalid() && DKind != GTaPDirectiveKind::GTaPD_entry) {
      return StmtError();
    }
    AStmt = Body.get();
  }

  // Build AST nodes through Sema.
  // For init directive, pass the parsed arguments
  if (DKind == GTaPDirectiveKind::GTaPD_init) {
    return getActions().GTaP().ActOnGTaPInitDirective(StartLoc, EndLoc, RuntimeType, FunctionName);
  }
  if (DKind == GTaPDirectiveKind::GTaPD_task) {
    return getActions().GTaP().ActOnGTaPTaskDirective(AStmt, StartLoc, EndLoc, QueueExpr);
  }
  if (DKind == GTaPDirectiveKind::GTaPD_taskwait) {
    return getActions().GTaP().ActOnGTaPTaskwaitDirective(StartLoc, EndLoc, QueueExpr);
  }
  return getActions().GTaP().ActOnGTaPExecutableDirective(DKind, AStmt, StartLoc, EndLoc, nullptr);
}
