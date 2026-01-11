// This file is made by maeda under gpu-task-parallelism project based on llvm-project.
// Referencing to clang/include/clang/Basic/OpenMPKinds.h

#ifndef LLVM_CLANG_BASIC_GTAPKINDS_H
#define LLVM_CLANG_BASIC_GTAPKINDS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

namespace clang {

/// gtap pragma directives.
enum class GTaPDirectiveKind : unsigned {
#define GTAP_DIRECTIVE(Name) GTaPD_##Name,
#include "clang/Basic/GTaPKinds.def"
  GTaPD_unknown
};

/// gtap pragma clauses.
enum class GTaPClauseKind : unsigned {
#define GTAP_CLAUSE(Name) GTaPC_##Name,
#include "clang/Basic/GTaPKinds.def"
  GTaPC_unknown
};

/// Return the directive kind for \param Str or GTaPD_unknown if not recognized.
inline GTaPDirectiveKind getGTaPDirectiveKind(llvm::StringRef Str) {
  return llvm::StringSwitch<GTaPDirectiveKind>(Str)
#define GTAP_DIRECTIVE(Name) .Case(#Name, GTaPDirectiveKind::GTaPD_##Name)
#include "clang/Basic/GTaPKinds.def"
      .Default(GTaPDirectiveKind::GTaPD_unknown);
}

/// Return the clause kind for \param Str or GTaPC_unknown if not recognized.
inline GTaPClauseKind getGTaPClauseKind(llvm::StringRef Str) {
  return llvm::StringSwitch<GTaPClauseKind>(Str)
#define GTAP_CLAUSE(Name) .Case(#Name, GTaPClauseKind::GTaPC_##Name)
#include "clang/Basic/GTaPKinds.def"
      .Default(GTaPClauseKind::GTaPC_unknown);
}

/// Return textual representation of directive \param Kind.
inline const char *getGTaPDirectiveName(GTaPDirectiveKind Kind) {
  switch (Kind) {
#define GTAP_DIRECTIVE(Name)  \
  case GTaPDirectiveKind::GTaPD_##Name:  \
    return #Name;
#include "clang/Basic/GTaPKinds.def"
  case GTaPDirectiveKind::GTaPD_unknown:
    break;
  }
  return "unknown";
}

/// Return textual representation of clause \param Kind.
inline const char *getGTaPClauseName(GTaPClauseKind Kind) {
  switch (Kind) {
#define GTAP_CLAUSE(Name)  \
  case GTaPClauseKind::GTaPC_##Name:  \
    return #Name;
#include "clang/Basic/GTaPKinds.def"
  case GTaPClauseKind::GTaPC_unknown:
    break;
  }
  return "unknown";
}

}
#endif
