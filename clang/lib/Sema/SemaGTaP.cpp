// This file is made by maeda under gpu-task-parallelism project based on llvm-project.
// Referencing to clang/lib/Sema/SemaOpenMP.cpp

#include "clang/Sema/SemaGTaP.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtGTaP.h"
#include "clang/Basic/AttrKinds.h"
#include "clang/Basic/GTaPKinds.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Sema.h"
#include "TreeTransform.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <string>
#include <unordered_map>

using namespace clang;

namespace {

struct GTaPExprBuilder {
  Sema &SemaRef;
  ASTContext &Ctx;
  ValueDecl *SelfParam;

  GTaPExprBuilder(Sema &S, ValueDecl *Self)
      : SemaRef(S), Ctx(S.Context), SelfParam(Self) {}

  Expr *buildSelfRef(SourceLocation Loc = SourceLocation()) const {
    return DeclRefExpr::Create(
        Ctx, NestedNameSpecifierLoc(), Loc,
        SelfParam, /*RefersToEnclosingVariableOrCapture=*/false,
        Loc, SelfParam->getType(), VK_LValue);
  }

  ExprResult buildFieldAccess(Expr *Base, bool IsArrow, FieldDecl *F,
                              SourceLocation Loc) {
    CXXScopeSpec SS;
    DeclarationNameInfo NameInfo(F->getDeclName(), Loc);
    return SemaRef.BuildMemberReferenceExpr(
        Base, Base->getType(), Loc, IsArrow, SS,
        SourceLocation(), nullptr, NameInfo, nullptr, nullptr);
  }
};

static bool isMacroDefined(Sema &S, StringRef Name) {
  Preprocessor &PP = S.getPreprocessor();
  return PP.isMacroDefined(Name);
}

static FunctionDecl *lookupRuntimeFunction(Sema &S, StringRef Name) {
  if (!S.TUScope)
    return nullptr;
  IdentifierInfo &II = S.Context.Idents.get(Name);
  LookupResult LR(S, &II, SourceLocation(), Sema::LookupOrdinaryName);
  if (!S.LookupName(LR, S.TUScope))
    return nullptr;
  for (NamedDecl *ND : LR) {
    if (auto *FD = dyn_cast<FunctionDecl>(ND))
      return FD;
  }
  return nullptr;
}

static FunctionDecl *requireRuntimeFunction(Sema &S, StringRef Name, SourceLocation Loc) {
  if (FunctionDecl *FD = lookupRuntimeFunction(S, Name))
    return FD;
  S.Diag(Loc, diag::err_gtap_runtime_function_not_found) << Name;
  return nullptr;
}

static QualType lookupNamedType(Sema &S, StringRef Name) {
  if (!S.TUScope)
    return QualType();
  IdentifierInfo &II = S.Context.Idents.get(Name);
  LookupResult LR(S, &II, SourceLocation(), Sema::LookupOrdinaryName);
  if (!S.LookupName(LR, S.TUScope))
    return QualType();
  for (NamedDecl *ND : LR) {
    if (auto *TD = dyn_cast<TypeDecl>(ND))
      return S.Context.getTypeDeclType(TD);
  }
  return QualType();
}

static uint64_t getMacroIntegerValue(Sema &S, StringRef Name, uint64_t Default) {
  Preprocessor &PP = S.getPreprocessor();
  IdentifierInfo *II = PP.getIdentifierInfo(Name);
  if (!II)
    return Default;
  MacroInfo *MI = PP.getMacroInfo(II);
  if (!MI || MI->getNumTokens() != 1)
    return Default;

  const Token &Tok = MI->tokens().front();
  if (!Tok.is(tok::numeric_constant))
    return Default;

  SmallString<32> Spelling;
  bool Invalid = false;
  StringRef Text = PP.getSpelling(Tok, Spelling, &Invalid);
  uint64_t Value = Default;
  if (!Invalid && !Text.getAsInteger(0, Value))
    return Value;
  return Default;
}

static Expr *buildThreadIdxXExpr(Sema &S, SourceLocation Loc) {
  ASTContext &Ctx = S.getASTContext();
  IdentifierInfo &ThreadIdxId = Ctx.Idents.get("threadIdx");
  DeclContext::lookup_result ThreadIdxLookup =
      Ctx.getTranslationUnitDecl()->lookup(&ThreadIdxId);
  if (ThreadIdxLookup.empty())
    return nullptr;

  auto *ThreadIdxVar = dyn_cast<VarDecl>(ThreadIdxLookup.front());
  if (!ThreadIdxVar)
    return nullptr;

  Expr *ThreadIdxRef = DeclRefExpr::Create(
      Ctx, NestedNameSpecifierLoc(), Loc, ThreadIdxVar,
      false, Loc, ThreadIdxVar->getType(), VK_LValue);

  CXXScopeSpec SS;
  IdentifierInfo &XId = Ctx.Idents.get("x");
  DeclarationNameInfo XNameInfo(&XId, Loc);
  ExprResult ThreadIdxX = S.BuildMemberReferenceExpr(
      ThreadIdxRef, ThreadIdxVar->getType(), Loc, false, SS,
      SourceLocation(), nullptr, XNameInfo, nullptr, nullptr);
  return ThreadIdxX.isInvalid() ? nullptr : ThreadIdxX.get();
}

static bool getGTaPMaxTaskSizeFromConstexpr(Sema &S, SourceLocation Loc,
                                          uint64_t &Out) {
  ASTContext &Ctx = S.getASTContext();
  if (!S.TUScope) return false;

  IdentifierInfo &II = Ctx.Idents.get("__gtap_max_task_size");
  LookupResult LR(S, &II, Loc, Sema::LookupOrdinaryName);
  if (!S.LookupName(LR, S.TUScope)) return false;

  for (NamedDecl *ND : LR) {
    if (auto *VD = dyn_cast<VarDecl>(ND)) {
      const Expr *Init = VD->getAnyInitializer();
      if (!Init) continue;

      Expr::EvalResult ER;
      if (!Init->EvaluateAsInt(ER, Ctx)) continue;
      if (!ER.Val.isInt()) continue;

      llvm::APSInt V = ER.Val.getInt();
      if (V.isSigned() && V.isNegative()) continue;

      Out = V.getZExtValue();
      return true;
    }

    if (auto *ED = dyn_cast<EnumConstantDecl>(ND)) {
      llvm::APSInt V = ED->getInitVal();
      if (V.isSigned() && V.isNegative()) continue;

      Out = V.getZExtValue();
      return true;
    }
  }
  return false;
}

static void checkTaskRecordSizeOrDiag(Sema &S, SourceLocation Loc, QualType TaskRecordTy) {
  ASTContext &Ctx = S.getASTContext();

  uint64_t MaxBytes = 0;
  if (!getGTaPMaxTaskSizeFromConstexpr(S, Loc, MaxBytes)) {
    S.Diag(Loc, diag::err_gtap_max_task_size_not_found);
    return;
  }

  if (S.RequireCompleteType(Loc, TaskRecordTy, diag::err_typecheck_incomplete_tag))
    return;

  uint64_t Bytes = Ctx.getTypeSizeInChars(TaskRecordTy).getQuantity();
  S.GTaP().noteTaskRecordSize(Bytes);
  if (Bytes > MaxBytes) {
    S.Diag(Loc, diag::err_gtap_task_record_too_large) << Bytes << MaxBytes;
  }
}

static bool checkTaskDataFieldTypeOrDiag(Sema &S, SourceLocation Loc,
                                         StringRef FieldName, QualType FieldTy) {
  if (FieldTy.isNull())
    return false;
  if (FieldTy.isTriviallyCopyableType(S.getASTContext()))
    return true;
  S.Diag(Loc, diag::err_gtap_task_data_non_trivially_copyable)
      << FieldName << FieldTy;
  return false;
}

static bool stmtContainsGTaPTaskwait(Stmt *S) {
  struct Finder : RecursiveASTVisitor<Finder> {
    bool Found = false;
    bool VisitGTaPTaskwaitDirective(GTaPTaskwaitDirective *) {
      Found = true;
      return false;
    }
  } F;
  F.TraverseStmt(S);
  return F.Found;
}

static bool checkNoTaskwaitInSwitchOrDiag(Sema &S, Stmt *Body) {
  struct Checker : RecursiveASTVisitor<Checker> {
    Sema &S;
    bool Valid = true;

    explicit Checker(Sema &S) : S(S) {}

    bool TraverseSwitchStmt(SwitchStmt *SS) {
      if (SS && stmtContainsGTaPTaskwait(SS->getBody())) {
        S.Diag(SS->getBeginLoc(), diag::err_gtap_taskwait_in_switch);
        Valid = false;
        return false;
      }
      return RecursiveASTVisitor<Checker>::TraverseSwitchStmt(SS);
    }
  } C(S);
  C.TraverseStmt(Body);
  return C.Valid;
}

static RecordDecl *createTaskDataRecord(Sema &S, FunctionDecl *FD,
                                        GTaPTaskFunctionInfo &TaskInfo,
                                        llvm::DenseMap<const ValueDecl *, FieldDecl *> &FieldMap) {
  ASTContext &Ctx = S.getASTContext();
  FieldMap.clear();
  const bool IsBlockWorker = isMacroDefined(S, "__GTAP_WORKER_IS_BLOCK");
  const uint64_t WorkerSize = getMacroIntegerValue(S, "GTAP_BLOCK_SIZE", 1);

  if (TaskInfo.TaskRecordInvalid)
    return nullptr;

  auto addField = [&](RecordDecl *RD, StringRef Name, QualType QT) -> FieldDecl * {
    IdentifierInfo &FieldId = Ctx.Idents.get(Name);
    FieldDecl *Field = FieldDecl::Create(
        Ctx, RD, SourceLocation(), SourceLocation(), &FieldId, QT,
        Ctx.getTrivialTypeSourceInfo(QT), nullptr, false, ICIS_NoInit);
    RD->addDecl(Field);
    return Field;
  };

  if (!TaskInfo.TaskRecord) {
    std::string RecordName = FD->getName().str() + "_task_data";
    IdentifierInfo &RecordId = Ctx.Idents.get(RecordName);
    RecordDecl *RD = RecordDecl::Create(Ctx, TagDecl::TagKind::Struct,
                                        Ctx.getTranslationUnitDecl(),
                                        FD->getBeginLoc(), FD->getLocation(),
                                        &RecordId);
    RD->startDefinition();

    std::unordered_map<std::string, unsigned> UsedFieldNames;
    auto makeUniqueFieldName = [&](std::string Base) -> std::string {
      if (Base.empty())
        Base = "__gtap_field";

      unsigned &Next = UsedFieldNames[Base];
      if (Next == 0) {
        Next = 1;
        return Base;
      }

      std::string Candidate;
      do {
        Candidate = Base + "_" + std::to_string(Next++);
      } while (UsedFieldNames.find(Candidate) != UsedFieldNames.end());
      UsedFieldNames[Candidate] = 1;
      return Candidate;
    };

    TaskInfo.ParameterFields.clear();
    unsigned ParamIndex = 0;
    for (ParmVarDecl *Param : TaskInfo.Parameters) {
      if (!Param)
        continue;
      std::string FieldName =
          Param->getIdentifier()
              ? Param->getName().str()
              : ("__param_" + std::to_string(ParamIndex));
      ++ParamIndex;
      FieldName = makeUniqueFieldName(FieldName);
      if (!checkTaskDataFieldTypeOrDiag(S, Param->getLocation(), FieldName,
                                        Param->getType())) {
        TaskInfo.TaskRecordInvalid = true;
        return nullptr;
      }
      QualType FieldTy = Param->getType();
      const bool IsUniformBlockParameter =
          IsBlockWorker && FieldTy.isConstQualified();
      // A top-level const parameter cannot change during the task's lifetime,
      // so all CUDA threads in a block task may read the single value supplied
      // by the spawning thread.  Strip only the top-level qualifier from the
      // storage type so spawn/entry initialization remains assignable; nested
      // qualifiers such as the pointee const in `const T *const` are retained.
      if (IsUniformBlockParameter)
        FieldTy = FieldTy.getUnqualifiedType();
      // A block task is resumed collectively, so an ordinary CUDA function
      // parameter must retain one value per thread.  The argument expression
      // is evaluated once by the spawning thread; state 0 of the child then
      // expands that value into the other elements of this array.
      if (IsBlockWorker && !IsUniformBlockParameter) {
        FieldTy = Ctx.getConstantArrayType(
            FieldTy, llvm::APInt(64, WorkerSize), nullptr,
            ArraySizeModifier::Normal, 0);
      }
      FieldDecl *Field = addField(RD, FieldName, FieldTy);
      TaskInfo.ParameterFields.push_back(Field);
      FieldMap[dyn_cast<ValueDecl>(Param->getCanonicalDecl())] = Field;
    }

    TaskInfo.CapturedFields.clear();
    unsigned CaptureIndex = 0;
    for (VarDecl *VD : TaskInfo.CapturedVariables) {
      if (!VD)
        continue;
      std::string FieldName =
          VD->getIdentifier()
              ? ("__cap_" + VD->getName().str())
              : ("__cap_anon_" + std::to_string(CaptureIndex));
      ++CaptureIndex;
      FieldName = makeUniqueFieldName(FieldName);
      QualType FieldTy = VD->getType();
      if (!checkTaskDataFieldTypeOrDiag(S, VD->getLocation(), FieldName,
                                        FieldTy)) {
        TaskInfo.TaskRecordInvalid = true;
        return nullptr;
      }
      if (IsBlockWorker) {
        FieldTy = Ctx.getConstantArrayType(
            FieldTy, llvm::APInt(64, WorkerSize), nullptr,
            ArraySizeModifier::Normal, 0);
      }
      FieldDecl *Field = addField(RD, FieldName, FieldTy);
      TaskInfo.CapturedFields.push_back(Field);
      FieldMap[dyn_cast<ValueDecl>(VD->getCanonicalDecl())] = Field;
    }

    // Note: State is now managed by runtime's taskheader, not in task struct
    TaskInfo.StateField = nullptr;

    TaskInfo.ResultField = nullptr;
    TaskInfo.ResultDstField = nullptr;
    TaskInfo.SpawningThreadField = nullptr;
    // Block tasks are entered collectively, even though a task directive is
    // encountered by one CUDA thread.  Remember that thread so the callee can
    // expand its argument values into the per-thread parameter slots when it
    // first starts executing.
    if (IsBlockWorker)
      TaskInfo.SpawningThreadField =
          addField(RD, "__gtap_spawning_thread", Ctx.IntTy);
    if (!TaskInfo.ReturnType.isNull() && !TaskInfo.ReturnType->isVoidType()) {
      if (!checkTaskDataFieldTypeOrDiag(S, FD->getLocation(), "__gtap_result",
                                        TaskInfo.ReturnType)) {
        TaskInfo.TaskRecordInvalid = true;
        return nullptr;
      }
      TaskInfo.ResultField = addField(RD, "__gtap_result", TaskInfo.ReturnType);
      TaskInfo.ResultDstField = addField(
          RD, "__gtap_result_dst",
          Ctx.getPointerType(TaskInfo.ReturnType));
    }

    RD->completeDefinition();
    Ctx.getTranslationUnitDecl()->addDecl(RD);
    TaskInfo.TaskRecord = RD;
    QualType TaskRecordTy = Ctx.getTypeDeclType(cast<TypeDecl>(RD));
    checkTaskRecordSizeOrDiag(S, FD->getLocation(), TaskRecordTy);

    return RD;
  }

  auto populateParams = [&](auto &Decls, auto &Fields) {
    size_t Count = std::min(Decls.size(), Fields.size());
    for (size_t I = 0; I < Count; ++I) {
      if (Decls[I] && Fields[I])
        FieldMap[dyn_cast<ValueDecl>(Decls[I]->getCanonicalDecl())] = Fields[I];
    }
  };
  populateParams(TaskInfo.Parameters, TaskInfo.ParameterFields);
  populateParams(TaskInfo.CapturedVariables, TaskInfo.CapturedFields);
  if (TaskInfo.TaskRecord && !TaskInfo.TaskRecord->isCompleteDefinition())
    TaskInfo.TaskRecord->completeDefinition();

  return TaskInfo.TaskRecord;
}

class GTaPTaskBodyTransformer
    : public TreeTransform<GTaPTaskBodyTransformer> {
  using Base = TreeTransform<GTaPTaskBodyTransformer>;
  GTaPExprBuilder &B;
  ASTContext &Ctx;
  VarDecl *SelfDecl;
  llvm::DenseMap<const ValueDecl *, FieldDecl *> FieldMap;
  FieldDecl *ResultField;
  FieldDecl *ResultDstField;
  FieldDecl *SpawningThreadField;
  ParmVarDecl *TidParam;
  ParmVarDecl *CtxParam;
  FunctionDecl *FinishDecl;
  VarDecl *ChildCountVar;  // Function-level child_count variable
  bool IsBlockWorker;
  unsigned NextWaitId = 0;

  bool InTaskDirective;
public:
  GTaPTaskBodyTransformer(Sema &S, GTaPExprBuilder &B, VarDecl *SelfDecl,
                         llvm::DenseMap<const ValueDecl *, FieldDecl *> FieldMap,
                         FieldDecl *ResultField,
                         FieldDecl *ResultDstField,
                         FieldDecl *SpawningThreadField,
                         ParmVarDecl *TidParam, ParmVarDecl *CtxParam,
                         FunctionDecl *FinishDecl, VarDecl *ChildCountVar,
                         bool IsBlockWorker)
      : Base(S), B(B), Ctx(S.getASTContext()), SelfDecl(SelfDecl),
        FieldMap(std::move(FieldMap)), ResultField(ResultField),
        ResultDstField(ResultDstField),
        SpawningThreadField(SpawningThreadField),
        TidParam(TidParam), CtxParam(CtxParam),
        FinishDecl(FinishDecl), ChildCountVar(ChildCountVar),
        IsBlockWorker(IsBlockWorker), InTaskDirective(false) {}

  // Generic hooks: log every statement we are about to transform.
  // We must preserve both overloads from TreeTransform:
  //   TransformStmt(Stmt *)
  //   TransformStmt(Stmt *, StmtDiscardKind)
  using Base::TransformStmt;

  // StmtResult TransformStmt(Stmt *S) {
  //   if (S)
  //     llvm::errs() << "[GTaP][Sema][Body] TransformStmt(kind="
  //                  << S->getStmtClassName() << ")\n";
  //   else
  //     llvm::errs() << "[GTaP][Sema][Body] TransformStmt(kind=<null>)\n";
  //   return Base::TransformStmt(S);
  // }

  // StmtResult TransformStmt(Stmt *S,
  //                          typename Base::StmtDiscardKind SDK) {
  //   if (S)
  //     llvm::errs() << "[GTaP][Sema][Body] TransformStmt(kind="
  //                  << S->getStmtClassName() << ", SDK)\n";
  //   else
  //     llvm::errs() << "[GTaP][Sema][Body] TransformStmt(kind=<null>, SDK)\n";
  //   return Base::TransformStmt(S, SDK);
  // }

  ExprResult TransformDeclRefExpr(DeclRefExpr *DRE) {
    const ValueDecl *VD = DRE->getDecl();
    const auto *canonRef = dyn_cast<ValueDecl>(VD->getCanonicalDecl());
    // llvm::errs() << "[GTaP] TransformDeclRefExpr: "
    //              << canonRef->getName() << " decl=" << (const void*)canonRef << "\n";
    auto It = FieldMap.find(canonRef);
    if (It == FieldMap.end()) {
      // llvm::errs() << "[GTaP] FieldMap MISS in DRE: "
      //              << canonRef->getName() << " decl=" << (const void*)canonRef << "\n";
      return Base::TransformDeclRefExpr(DRE);
    }
    FieldDecl *Field = It->second;
    ExprResult MemberER = buildCapturedFieldAccess(Field, DRE->getExprLoc());
    if (MemberER.isInvalid()) return ExprError();
    Expr *Member = MemberER.get();

    // if (FieldMap.count(canonRef)) {
    //   llvm::errs() << "[GTaP] FieldMap HIT in DRE: "
    //                << canonRef->getName() << " decl=" << (const void*)canonRef << "\n";
    // }
    return Member;
  }

  ExprResult TransformCallExpr(CallExpr *Call) {
    if (!InTaskDirective) {
      if (const FunctionDecl *Callee = Call->getDirectCallee()) {
        if (Base::getSema().GTaP().isGTaPTaskFunction(Callee)) {
          Base::getSema().Diag(Call->getExprLoc(),
                              diag::err_gtap_direct_task_function_call)
              << Callee;
          return ExprError();
        }
      }
    }
    return Base::TransformCallExpr(Call);
  }

  // For debug; unnecessary any more
  // StmtResult TransformWhileStmt(WhileStmt *WS) {
  //   llvm::errs() << "[VR] TransformWhileStmt\n";
  //   return Base::TransformWhileStmt(WS);
  // }

  StmtResult TransformDeclStmt(DeclStmt *DS) {
    // llvm::errs() << "[VR] TransformDeclStmt\n";
  
    SmallVector<Stmt*, 4> Stmts;          // Return statements (DeclStmt + Assign)
    SmallVector<Decl*, 4> KeepDecls;      // For non-captured VarDecls to keep
  
    for (Decl *D : DS->decls()) {
      auto *VD = dyn_cast<VarDecl>(D);
      if (!VD) {
        return Base::TransformDeclStmt(DS);
      }
  
      const ValueDecl *Key = VD->getCanonicalDecl();
      FieldDecl *Field = FieldMap.lookup(Key);
  
      Expr *NewInitExpr = nullptr;
      if (VD->hasInit()) {
        ExprResult RHS = getDerived().TransformExpr(VD->getInit());
        if (RHS.isInvalid()) return StmtError();
        NewInitExpr = RHS.get();
      }
  
      if (Field) {
        // captured: remove the declaration and replace it with self->field = init;
        if (!NewInitExpr) {
          continue;
        }

        // Arrays are not assignable.  Keep the initialized local array as a
        // temporary and copy its complete object representation into the task
        // record.  Subsequent references are still rewritten to the field.
        if (Ctx.getAsConstantArrayType(VD->getType())) {
          if (NewInitExpr != VD->getInit())
            VD->setInit(NewInitExpr);
          KeepDecls.push_back(VD);

          ExprResult LHSER = buildCapturedFieldAccess(Field, SourceLocation());
          if (LHSER.isInvalid()) return StmtError();
          Expr *To = buildAddressAsVoidPointer(LHSER.get(), /*IsConst=*/false);
          Expr *From = buildAddressAsVoidPointer(
              buildDeclRefLValue(VD), /*IsConst=*/true);
          QualType SizeTy = Ctx.getSizeType();
          CharUnits ArraySize = Ctx.getTypeSizeInChars(VD->getType());
          Expr *Size = IntegerLiteral::Create(
              Ctx,
              llvm::APInt(Ctx.getTypeSize(SizeTy), ArraySize.getQuantity()),
              SizeTy, SourceLocation());
          Expr *CopyArgs[] = {To, From, Size};
          Stmts.push_back(SemaRef.BuildBuiltinCallExpr(
              SourceLocation(), Builtin::BI__builtin_memcpy, CopyArgs));
          continue;
        }

        ExprResult LHSER = buildCapturedFieldAccess(Field, SourceLocation());
        if (LHSER.isInvalid()) return StmtError();
        Expr *LHS = LHSER.get();
        ExprResult Assign = SemaRef.BuildBinOp(
            /*Scope=*/nullptr, SourceLocation(), BO_Assign, LHS, NewInitExpr);
        if (Assign.isInvalid()) return StmtError();
        Stmts.push_back(Assign.get());
      } else {
        if (VD->hasInit() && NewInitExpr && NewInitExpr != VD->getInit()) {
          VD->setInit(NewInitExpr);
        }
        KeepDecls.push_back(VD);
      }
    }
  
    // if there is at least one captured, build a CompoundStmt (DeclStmt + Assign)
    if (!Stmts.empty()) {
      if (!KeepDecls.empty()) {
        DeclGroupRef DG = DeclGroupRef::Create(Ctx, KeepDecls.data(), KeepDecls.size());
        Stmt *NewDeclStmt = new (Ctx) DeclStmt(DG, DS->getBeginLoc(), DS->getEndLoc());
        Stmts.insert(Stmts.begin(), NewDeclStmt);
      }
      return getDerived().RebuildCompoundStmt(
          SourceLocation(), Stmts, SourceLocation(), /*IsStmtExpr=*/false);
    }
  
    return StmtResult(DS);
  }

  StmtResult TransformReturnStmt(ReturnStmt *S) {
    SmallVector<Stmt *, 4> Sequence;
    if (ResultField && S->getRetValue()) {
      ExprResult Ret = getDerived().TransformExpr(S->getRetValue());
      if (Ret.isInvalid())
        return StmtError();
      
      // Convert lvalue to rvalue if necessary
      Expr *RHS = Ret.get();
      if (RHS->isLValue() || RHS->isXValue()) {
        RHS = ImplicitCastExpr::Create(
            Ctx, RHS->getType(), CK_LValueToRValue, RHS, nullptr, VK_PRValue,
            FPOptionsOverride());
      }
      
      if (!ResultDstField)
        return StmtError();
      ExprResult DstER = B.buildFieldAccess(
          B.buildSelfRef(), true, ResultDstField, SourceLocation());
      if (DstER.isInvalid())
        return StmtError();
      Expr *DstLHS = UnaryOperator::Create(
          Ctx, toRValue(DstER.get()), UO_Deref,
          ResultField->getType(), VK_LValue, OK_Ordinary,
          SourceLocation(), false, FPOptionsOverride());
      ExprResult StoreER = SemaRef.BuildBinOp(
          nullptr, SourceLocation(), BO_Assign, DstLHS, RHS);
      if (StoreER.isInvalid())
        return StmtError();
      Stmt *ResultWriteStmt = StoreER.get();

      const bool GuardResultWrite = SpawningThreadField != nullptr;
      if (GuardResultWrite) {
  
        Expr *Cond = nullptr;
        {
          IdentifierInfo &ThreadIdxId = Ctx.Idents.get("threadIdx");
          auto Lookup = Ctx.getTranslationUnitDecl()->lookup(&ThreadIdxId);
          VarDecl *ThreadIdxVar =
              Lookup.empty() ? nullptr : dyn_cast<VarDecl>(Lookup.front());
  
          if (ThreadIdxVar) {
            Expr *ThreadIdxRef = DeclRefExpr::Create(
                Ctx, NestedNameSpecifierLoc(), SourceLocation(), ThreadIdxVar,
                false, SourceLocation(), ThreadIdxVar->getType(), VK_LValue);
  
            CXXScopeSpec SS;
            IdentifierInfo &XId = Ctx.Idents.get("x");
            DeclarationNameInfo XNameInfo(&XId, SourceLocation());
  
            ExprResult ThreadIdxX =
                SemaRef.BuildMemberReferenceExpr(ThreadIdxRef, ThreadIdxVar->getType(),
                                                 SourceLocation(), /*IsArrow*/false, SS,
                                                 SourceLocation(), nullptr, XNameInfo,
                                                 nullptr, nullptr);
  
            if (!ThreadIdxX.isInvalid()) {
              ExprResult SpawningThreadER = B.buildFieldAccess(
                  B.buildSelfRef(), true, SpawningThreadField,
                  SourceLocation());
              if (SpawningThreadER.isInvalid())
                return StmtError();
              Expr *SpawningThread = SpawningThreadER.get();
              if (SpawningThread->isGLValue())
                SpawningThread = ImplicitCastExpr::Create(
                    Ctx, SpawningThread->getType(), CK_LValueToRValue,
                    SpawningThread, nullptr, VK_PRValue,
                    FPOptionsOverride());
  
              ExprResult Eq = SemaRef.BuildBinOp(
                  nullptr, SourceLocation(), BO_EQ, ThreadIdxX.get(),
                  SpawningThread);
              if (!Eq.isInvalid()) Cond = Eq.get();
            }
          }
        }
  
        if (Cond) {
          Stmt *ThenS = ResultWriteStmt;
          IfStmt *If = IfStmt::Create(
              Ctx, SourceLocation(),
              IfStatementKind::Ordinary,
              nullptr, nullptr, Cond,
              SourceLocation(), SourceLocation(),
              ThenS,
              SourceLocation(), nullptr);
  
          Sequence.push_back(If);
        } else {
          Sequence.push_back(ResultWriteStmt);
        }
      } else {
        Sequence.push_back(ResultWriteStmt);
      }
    }

    if (FinishDecl) {
      // __gtap_finish_task(int tid, TaskContext* ctx)
      SmallVector<Expr *, 2> Args;
      Args.push_back(buildParamRValue(TidParam));
      Args.push_back(buildParamRValue(CtxParam));
      Sema &SemaRef = Base::getSema();
      ExprResult Callee = SemaRef.BuildDeclRefExpr(
          FinishDecl, FinishDecl->getType(), VK_LValue, SourceLocation());
      if (Callee.isInvalid())
        return StmtError();
      ExprResult Call = SemaRef.BuildCallExpr(
          /*Scope=*/nullptr, Callee.get(), SourceLocation(), Args,
          SourceLocation());
      if (Call.isInvalid())
        return StmtError();
      Sequence.push_back(Call.get());
    }

    Sequence.push_back(
        ReturnStmt::Create(Ctx, SourceLocation(), nullptr, nullptr));
    return StmtResult(CompoundStmt::Create(
        Ctx, Sequence, FPOptionsOverride(), SourceLocation(), SourceLocation()));
  }
  
  StmtResult TransformGTaPTaskDirective(GTaPTaskDirective *Dir) {
    if (!Dir->hasAssociatedStmt())
      return StmtResult(new (Ctx) NullStmt(SourceLocation()));
    
    Stmt *OriginalAssociatedStmt = Dir->getAssociatedStmt();
    
    // Set flag to indicate we're inside a task directive BEFORE transforming
    // This prevents TransformCallExpr from transforming GTaP function calls
    // that should be handled here at AST level
    bool OldInTaskDirective = InTaskDirective;
    InTaskDirective = true;
    
    // Handle pattern: a = fib(n-1) inside #pragma gtap task
    // We need to extract the function call BEFORE transforming the statement
    // Store the result variable for later assignment after taskwait
    CallExpr *OriginalCall = nullptr;
    Expr *ResultLHS = nullptr;
    
    if (auto *BinOp = dyn_cast<BinaryOperator>(OriginalAssociatedStmt)) {
      if (BinOp->getOpcode() == BO_Assign) {
        Expr *RHS = BinOp->getRHS();
        if (auto *Call = dyn_cast<CallExpr>(RHS->IgnoreParenImpCasts())) {
          const FunctionDecl *Callee = Call->getDirectCallee();
          if (Callee && Callee->hasAttr<GTaPFunctionAttr>()) {
            // Extract result variable and original call BEFORE transformation
            ResultLHS = BinOp->getLHS();
            OriginalCall = Call;
          }
        }
      }
    }

    if (!OriginalCall) {
      if (auto *E = dyn_cast<Expr>(OriginalAssociatedStmt)) {
        if (auto *Call = dyn_cast<CallExpr>(E->IgnoreParenImpCasts()))
          OriginalCall = Call;
      }
    }

    if (OriginalCall) {
      if (auto *Callee = OriginalCall->getDirectCallee())
        if (!Callee->hasAttr<GTaPFunctionAttr>())
          OriginalCall = nullptr;
    }

    if (!OriginalCall) {
      Base::getSema().Diag(Dir->getBeginLoc(), diag::err_gtap_task_invalid_statement);
      InTaskDirective = OldInTaskDirective;
      return StmtError();
    }
    
    // If we found a GTaP function call, transform it to task spawning code
    {
      const FunctionDecl *Callee = OriginalCall->getDirectCallee();
      
      // This is a GTaP function call inside #pragma gtap task
      // Create task data structure and populate fields with arguments
      // Then create CallExpr with the correct signature: void (void*, int, TaskContext*)
      
      // Get the callee's task record and field map
      // Access SemaGTaP through Sema's GTaP() method
      Sema &S = Base::getSema();
      SemaGTaP &SemaGTaPRef = S.GTaP();
      
      // Get or create task info for the callee
      FunctionDecl *CalleeFD = const_cast<FunctionDecl *>(Callee);
      GTaPTaskFunctionInfo &CalleeTaskInfo = SemaGTaPRef.getCachedTaskInfo(CalleeFD);
      
      // Create task record and field map if needed
      llvm::DenseMap<const ValueDecl *, FieldDecl *> CalleeFieldMap;
      RecordDecl *CalleeTaskRecord = createTaskDataRecord(S, CalleeFD, CalleeTaskInfo, CalleeFieldMap);
      if (!CalleeTaskRecord) {
        InTaskDirective = OldInTaskDirective;
        return StmtError();
      }
      
      QualType TaskRecordTy = Ctx.getTypeDeclType(cast<TypeDecl>(CalleeTaskRecord));
      const bool IsBlockWorker =
          isMacroDefined(Base::getSema(), "__GTAP_WORKER_IS_BLOCK");
      QualType VoidTy = Ctx.VoidTy;
      QualType VoidPtrTy = Ctx.getPointerType(Ctx.VoidTy);
      QualType IntTy = Ctx.IntTy;
      QualType IntPtrTy = Ctx.getPointerType(IntTy);
      QualType TaskCtxPtrTy = VoidPtrTy;
      if (QualType TaskCtxTy = lookupNamedType(Base::getSema(), "TaskContext");
          !TaskCtxTy.isNull()) {
        TaskCtxPtrTy = Ctx.getPointerType(TaskCtxTy);
      }
      QualType TaskPtrTy = Ctx.getPointerType(TaskRecordTy);
      
      // Get the current function's DeclContext from SelfDecl
      DeclContext *CurrentDC = SelfDecl->getDeclContext();
      if (!CurrentDC) {
        InTaskDirective = OldInTaskDirective;
        return StmtError();
      }
      
      SmallVector<Stmt *, 8> TaskStmts;
      
      // Step 1: Call __gtap_spawn_task(ctx, self_tid, child_count, func, queue_idx) -> void*
      // This allocates task ID, sets up TaskHeader, and returns task data pointer
      FunctionDecl *SpawnTaskFunc = requireRuntimeFunction(
          Base::getSema(), "__gtap_spawn_task", Dir->getBeginLoc());
      if (!SpawnTaskFunc) {
        InTaskDirective = OldInTaskDirective;
        return StmtError();
      }
      // Build queue_idx argument
      Expr *QueueArg = nullptr;
      if (Expr *QE = Dir->getQueueExpr()) {
        ExprResult TQ = getDerived().TransformExpr(QE);
        if (TQ.isInvalid()) return StmtError();
        Expr *E = TQ.get();
        if (!E->getType()->isIntegerType()) {
          // diag and fallback to 0
        }
        QueueArg = ImplicitCastExpr::Create(
            Ctx, IntTy, CK_IntegralCast, toRValue(E), nullptr, VK_PRValue, FPOptionsOverride());
      }
      if (!QueueArg) {
        QueueArg = IntegerLiteral::Create(
            Ctx, llvm::APInt(Ctx.getIntWidth(IntTy), 0), IntTy, SourceLocation());
      }
      
      // Build func_ptr argument
      auto &TI = SemaGTaPRef.getCachedTaskInfo(CalleeFD);
      FunctionDecl *StateMachineFD = TI.StateMachineFD;
      if (!StateMachineFD) {
        StateMachineFD = SemaGTaPRef.getOrCreateStateMachineFunction(CalleeFD, VoidTy, VoidPtrTy, IntTy, TaskCtxPtrTy);
      }
      
      ExprResult StateMachineRefER = S.BuildDeclRefExpr(StateMachineFD, StateMachineFD->getType(), VK_LValue, SourceLocation());
      Expr *StateMachineRef = StateMachineRefER.get();
      
      QualType StateMachinePtrTy = Ctx.getPointerType(StateMachineFD->getType());
      Expr *StateMachinePtr = ImplicitCastExpr::Create(
          Ctx, StateMachinePtrTy, CK_FunctionToPointerDecay, StateMachineRef, nullptr, VK_PRValue, FPOptionsOverride());
      
      Expr *ChildCountArg = nullptr;
      if (IsBlockWorker) {
        ChildCountArg = CStyleCastExpr::Create(
            Ctx, IntPtrTy, VK_PRValue, CK_NullToPointer,
            IntegerLiteral::Create(Ctx, llvm::APInt(64, 0), Ctx.IntTy,
                                   SourceLocation()),
            nullptr, FPOptionsOverride(), Ctx.getTrivialTypeSourceInfo(IntPtrTy),
            SourceLocation(), SourceLocation());
      } else {
        Expr *ChildCountRef = DeclRefExpr::Create(
            Ctx, NestedNameSpecifierLoc(), SourceLocation(), ChildCountVar,
            false, SourceLocation(), IntTy, VK_LValue);
        ChildCountArg = UnaryOperator::Create(
            Ctx, ChildCountRef, UO_AddrOf, IntPtrTy, VK_PRValue,
            OK_Ordinary, SourceLocation(), false, FPOptionsOverride());
      }

      // Build __gtap_spawn_task call
      // Signature: void* __gtap_spawn_task(TaskContext* ctx, int self_tid,
      //                                    int* child_count, void (*func)(...),
      //                                    int queue_idx)
      SmallVector<Expr *, 5> SpawnArgs;
      SpawnArgs.push_back(buildParamRValue(CtxParam));      // ctx
      SpawnArgs.push_back(buildParamRValue(TidParam));      // self_tid
      SpawnArgs.push_back(ChildCountArg);                    // child_count
      SpawnArgs.push_back(StateMachinePtr);                  // func
      SpawnArgs.push_back(QueueArg);                         // queue_idx
      
      ExprResult SpawnCallee = S.BuildDeclRefExpr(
          SpawnTaskFunc, SpawnTaskFunc->getType(), VK_LValue, SourceLocation());
      if (SpawnCallee.isInvalid())
        return StmtError();
      
      ExprResult SpawnCallER = S.BuildCallExpr(
          /*Scope=*/nullptr, SpawnCallee.get(), SourceLocation(),
          SpawnArgs, SourceLocation());
      if (SpawnCallER.isInvalid())
        return StmtError();
      Expr *SpawnCall = SpawnCallER.get();
      
      // Cast void* return value to TaskRecordTy*
      Expr *TaskPtrInit = CStyleCastExpr::Create(
          Ctx, TaskPtrTy, VK_PRValue, CK_BitCast, SpawnCall,
          nullptr, FPOptionsOverride(),
          Ctx.getTrivialTypeSourceInfo(TaskPtrTy),
          SourceLocation(), SourceLocation());
      
      // Declare: TaskRecordTy* __gtap_task_ptr = (TaskRecordTy*)__gtap_spawn_task(...);
      std::string TaskPtrName = "__gtap_task_ptr_" + Callee->getName().str() + "_" + 
                               std::to_string(reinterpret_cast<uintptr_t>(OriginalCall));
      IdentifierInfo &TaskPtrId = Ctx.Idents.get(TaskPtrName);
      VarDecl *TaskPtrVar = VarDecl::Create(
          Ctx, CurrentDC, SourceLocation(), SourceLocation(),
          &TaskPtrId, TaskPtrTy, Ctx.getTrivialTypeSourceInfo(TaskPtrTy),
          SC_None);
      TaskPtrVar->setInit(TaskPtrInit);
      
      DeclStmt *TaskPtrDecl = new (Ctx) DeclStmt(
          DeclGroupRef(TaskPtrVar), SourceLocation(), SourceLocation());
      TaskStmts.push_back(TaskPtrDecl);

      if (CalleeTaskInfo.SpawningThreadField) {
        Expr *ThreadIdxX = buildThreadIdxXExpr(S, SourceLocation());
        if (!ThreadIdxX)
          return StmtError();
        Expr *TaskPtrRef = DeclRefExpr::Create(
            Ctx, NestedNameSpecifierLoc(), SourceLocation(), TaskPtrVar,
            false, SourceLocation(), TaskPtrTy, VK_LValue);
        Expr *TaskPtrRValue = ImplicitCastExpr::Create(
            Ctx, TaskPtrTy, CK_LValueToRValue, TaskPtrRef, nullptr,
            VK_PRValue, FPOptionsOverride());
        ExprResult SpawningThreadFieldER = B.buildFieldAccess(
            TaskPtrRValue, /*IsArrow=*/true,
            CalleeTaskInfo.SpawningThreadField, SourceLocation());
        if (SpawningThreadFieldER.isInvalid())
          return StmtError();
        ExprResult AssignER = SemaRef.BuildBinOp(
            nullptr, SourceLocation(), BO_Assign,
            SpawningThreadFieldER.get(), ThreadIdxX);
        if (AssignER.isInvalid())
          return StmtError();
        TaskStmts.push_back(AssignER.get());
      }

      if (ResultLHS && CalleeTaskInfo.ResultDstField) {
        ExprResult LHSER = getDerived().TransformExpr(ResultLHS);
        if (LHSER.isInvalid())
          return StmtError();
        Expr *LHS = LHSER.get();
        QualType LHSAddrTy = Ctx.getPointerType(LHS->getType());
        Expr *LHSAddr = UnaryOperator::Create(
            Ctx, LHS, UO_AddrOf, LHSAddrTy, VK_PRValue,
            OK_Ordinary, SourceLocation(), false, FPOptionsOverride());
        Expr *TaskPtrRef = DeclRefExpr::Create(
            Ctx, NestedNameSpecifierLoc(), SourceLocation(), TaskPtrVar,
            false, SourceLocation(), TaskPtrTy, VK_LValue);
        Expr *TaskPtrRValue = ImplicitCastExpr::Create(
            Ctx, TaskPtrTy, CK_LValueToRValue, TaskPtrRef, nullptr,
            VK_PRValue, FPOptionsOverride());
        ExprResult DstFieldER = B.buildFieldAccess(
            TaskPtrRValue, true, CalleeTaskInfo.ResultDstField,
            SourceLocation());
        if (DstFieldER.isInvalid())
          return StmtError();
        ExprResult DstAssignER = SemaRef.BuildBinOp(
            nullptr, SourceLocation(), BO_Assign,
            DstFieldER.get(), LHSAddr);
        if (DstAssignER.isInvalid())
          return StmtError();
        TaskStmts.push_back(DstAssignER.get());
      }
      
      // Create TaskPtrRef for field access (use arrow operator since it's a pointer)
      auto buildTaskPtrRef = [&]() -> Expr* {
        Expr *PtrRef = DeclRefExpr::Create(
            Ctx, NestedNameSpecifierLoc(), SourceLocation(), TaskPtrVar,
            false, SourceLocation(), TaskPtrTy, VK_LValue);
        return ImplicitCastExpr::Create(
            Ctx, TaskPtrTy, CK_LValueToRValue, PtrRef, nullptr, VK_PRValue, FPOptionsOverride());
      };
      
      // Step 2: Store arguments to task data fields via returned pointer
      unsigned ArgIndex = 0;
      for (Expr *Arg : OriginalCall->arguments()) {
        ExprResult TransformedArg = getDerived().TransformExpr(Arg);
        if (TransformedArg.isInvalid())
          return StmtError();
        
        // Assign to corresponding parameter field via pointer (arrow access)
        if (ArgIndex < CalleeTaskInfo.Parameters.size()) {
          ParmVarDecl *Param = CalleeTaskInfo.Parameters[ArgIndex];
          if (Param) {
            const ValueDecl *Key = cast<ValueDecl>(Param->getCanonicalDecl());
            auto FieldIt = CalleeFieldMap.find(Key);
            if (FieldIt != CalleeFieldMap.end()) {
              FieldDecl *Field = FieldIt->second;
              // Use arrow access (IsArrow=true) since TaskPtrVar is a pointer
              ExprResult FieldRefER = B.buildFieldAccess(buildTaskPtrRef(), /*IsArrow=*/true, Field, SourceLocation());
              if (FieldRefER.isInvalid()) return StmtError();
              Expr *FieldRef = FieldRefER.get();
              const bool IsPerThreadBlockField =
                  IsBlockWorker && Ctx.getAsConstantArrayType(Field->getType());
              if (!IsPerThreadBlockField) {
                ExprResult AssignER = SemaRef.BuildBinOp(
                    nullptr, SourceLocation(), BO_Assign, FieldRef,
                    TransformedArg.get());
                if (AssignER.isInvalid()) return StmtError();
                TaskStmts.push_back(AssignER.get());
              } else {
                // Store only the spawning thread's parameter value.  State 0
                // of the child expands this value into the other per-thread
                // slots collectively.
                Expr *ThreadIdxX = buildThreadIdxXExpr(
                    SemaRef, SourceLocation());
                if (!ThreadIdxX)
                  return StmtError();
                Expr *Indices[] = {ThreadIdxX};
                ExprResult Element = SemaRef.ActOnArraySubscriptExpr(
                    nullptr, FieldRef, SourceLocation(),
                    MultiExprArg(Indices, 1), SourceLocation());
                if (Element.isInvalid())
                  return StmtError();
                ExprResult Assign = SemaRef.BuildBinOp(
                    nullptr, SourceLocation(), BO_Assign, Element.get(),
                    TransformedArg.get());
                if (Assign.isInvalid())
                  return StmtError();
                TaskStmts.push_back(Assign.get());
              }
            }
          }
        }
        ++ArgIndex;
      }
      
      // A discarded result is written to the child's own result field.
      // Assigned results have already installed the parent's destination.
      if (CalleeTaskInfo.ResultDstField && CalleeTaskInfo.ResultField &&
          !ResultLHS) {
        ExprResult DstFieldER = B.buildFieldAccess(
            buildTaskPtrRef(), /*IsArrow=*/true,
            CalleeTaskInfo.ResultDstField, SourceLocation());
        ExprResult ResultFieldER = B.buildFieldAccess(
            buildTaskPtrRef(), /*IsArrow=*/true,
            CalleeTaskInfo.ResultField, SourceLocation());
        if (DstFieldER.isInvalid() || ResultFieldER.isInvalid())
          return StmtError();
        Expr *OwnResultAddr = UnaryOperator::Create(
            Ctx, ResultFieldER.get(), UO_AddrOf,
            CalleeTaskInfo.ResultDstField->getType(), VK_PRValue,
            OK_Ordinary, SourceLocation(), false, FPOptionsOverride());
        ExprResult AssignER = SemaRef.BuildBinOp(
            nullptr, SourceLocation(), BO_Assign,
            DstFieldER.get(), OwnResultAddr);
        if (AssignER.isInvalid())
          return StmtError();
        TaskStmts.push_back(AssignER.get());
      }

      // Create compound statement
      Stmt *TaskCompound = CompoundStmt::Create(
          Ctx, TaskStmts, FPOptionsOverride(),
          SourceLocation(), SourceLocation());
      
      InTaskDirective = OldInTaskDirective;
      
      // Return the compound statement directly (no need to wrap in GTaPTaskDirective)
      return StmtResult(TaskCompound);
  }
  }

  StmtResult TransformGTaPTaskwaitDirective(GTaPTaskwaitDirective *Dir) {
    Expr *NewQueueExpr = nullptr;
    if (Expr *QE = Dir->getQueueExpr()) {
      ExprResult R = getDerived().TransformExpr(QE);
      if (R.isInvalid()) return StmtError();
      NewQueueExpr = R.get();
    }
    Dir->setQueueExpr(NewQueueExpr);
    unsigned id = NextWaitId++;
    Dir->setWaitId(id);
    return StmtResult(Dir);
  }

private:
  Expr *buildAddressAsVoidPointer(Expr *E, bool IsConst) const {
    QualType PointerTy = Ctx.getPointerType(E->getType());
    Expr *Address = UnaryOperator::Create(
        Ctx, E, UO_AddrOf, PointerTy, VK_PRValue, OK_Ordinary,
        SourceLocation(), false, SemaRef.CurFPFeatureOverrides());
    QualType VoidTy = IsConst ? Ctx.getConstType(Ctx.VoidTy) : Ctx.VoidTy;
    return ImplicitCastExpr::Create(
        Ctx, Ctx.getPointerType(VoidTy), CK_BitCast, Address, nullptr,
        VK_PRValue, FPOptionsOverride());
  }

  Expr *buildDeclRefLValue(ValueDecl *D) const {
    return DeclRefExpr::Create(Ctx, NestedNameSpecifierLoc(), SourceLocation(),
                               D, /*RefersToEnclosingVariableOrCapture=*/false,
                               SourceLocation(), D->getType(), VK_LValue);
  };
  
  Expr *toRValue(Expr *E) const {
    return ImplicitCastExpr::Create(Ctx, E->getType(), CK_LValueToRValue,
                                    E, nullptr, VK_PRValue, FPOptionsOverride());
  };
  
  Expr *buildParamRValue(ParmVarDecl *P) const {
    if (!P)
      return IntegerLiteral::Create(Ctx, llvm::APInt(Ctx.getIntWidth(Ctx.IntTy), 0),
                                    Ctx.IntTy, SourceLocation());
    return toRValue(buildDeclRefLValue(P));
  };

  ExprResult buildCapturedFieldAccess(FieldDecl *Field,
                                      SourceLocation Loc = SourceLocation()) {
    ExprResult MemberER = B.buildFieldAccess(B.buildSelfRef(), true, Field, Loc);
    if (MemberER.isInvalid())
      return ExprError();

    Expr *Member = MemberER.get();
    // Block-mode captures have an extra outer [GTAP_BLOCK_SIZE] dimension;
    // select the current thread's object.  A source-level array in thread mode
    // is the captured object itself and must remain an array lvalue.
    if (!IsBlockWorker || !Ctx.getAsConstantArrayType(Field->getType()))
      return Member;

    Expr *ThreadIdxX = buildThreadIdxXExpr(SemaRef, Loc);
    if (!ThreadIdxX)
      return ExprError();

    Expr *Args[] = {ThreadIdxX};
    return SemaRef.ActOnArraySubscriptExpr(
        nullptr, Member, Loc, MultiExprArg(Args, 1), Loc);
  }

};
} // namespace

SemaGTaP::SemaGTaP(Sema &S) : SemaBase(S) {}

ASTContext &SemaGTaP::getASTContext() { return SemaRef.getASTContext(); }

void SemaGTaP::noteTaskRecordSize(uint64_t Bytes) {
  if (Bytes == 0)
    Bytes = 1;
  if (Bytes <= AutoTaskDataSize && AutoTaskDataSizeDecl)
    return;

  AutoTaskDataSize = std::max(AutoTaskDataSize, Bytes);

  ASTContext &Ctx = getASTContext();
  IdentifierInfo &II = Ctx.Idents.get("__gtap_auto_task_data_size");
  QualType SizeTy = Ctx.getSizeType();
  QualType ConstSizeTy = Ctx.getConstType(SizeTy);
  Expr *Init = IntegerLiteral::Create(
      Ctx, llvm::APInt(Ctx.getTypeSize(SizeTy), AutoTaskDataSize),
      SizeTy, SourceLocation());

  if (!AutoTaskDataSizeDecl) {
    if (SemaRef.TUScope) {
      LookupResult LR(SemaRef, &II, SourceLocation(), Sema::LookupOrdinaryName);
      if (SemaRef.LookupName(LR, SemaRef.TUScope)) {
        for (NamedDecl *ND : LR) {
          if (auto *VD = dyn_cast<VarDecl>(ND)) {
            AutoTaskDataSizeDecl = VD;
            break;
          }
        }
      }
    }

    if (!AutoTaskDataSizeDecl) {
      AutoTaskDataSizeDecl = VarDecl::Create(
          Ctx, Ctx.getTranslationUnitDecl(), SourceLocation(), SourceLocation(),
          &II, ConstSizeTy, Ctx.getTrivialTypeSourceInfo(ConstSizeTy),
          SC_Extern);
      Ctx.getTranslationUnitDecl()->addDecl(AutoTaskDataSizeDecl);
    }
  }

  AutoTaskDataSizeDecl->setInit(Init);
}

StmtResult SemaGTaP::ActOnGTaPExecutableDirective(GTaPDirectiveKind DKind,
                                                Stmt *AStmt,
                                                SourceLocation StartLoc,
                                                SourceLocation EndLoc,
                                                Expr *QueueExpr) {
  switch (DKind) {
  case GTaPDirectiveKind::GTaPD_task:
    return ActOnGTaPTaskDirective(AStmt, StartLoc, EndLoc, QueueExpr);
  case GTaPDirectiveKind::GTaPD_taskwait:
    return ActOnGTaPTaskwaitDirective(StartLoc, EndLoc, QueueExpr);
  case GTaPDirectiveKind::GTaPD_init:
    // init directive is handled separately in ParseGTaPExecutableDirective
    // because it requires arguments (runtime type and function name)
    llvm_unreachable("init directive should not reach ActOnGTaPExecutableDirective");
  case GTaPDirectiveKind::GTaPD_entry:
    return ActOnGTaPEntryDirective(StartLoc, EndLoc, AStmt);
  case GTaPDirectiveKind::GTaPD_function:
    llvm_unreachable("function directive should not reach ActOnGTaPExecutableDirective");
  case GTaPDirectiveKind::GTaPD_unknown:
    llvm_unreachable("Unknown directive kind");
  }
  llvm_unreachable("Unhandled directive kind");
}

static Expr *defaultQueueExpr(ASTContext &Ctx, SourceLocation Loc) {
  return IntegerLiteral::Create(Ctx, llvm::APInt(/*numBits*/32, /*val*/0),
                                Ctx.IntTy, Loc);
}

static bool diagnoseNestedGTaPTaskFunctionCalls(Sema &S, Stmt *Root,
                                                const CallExpr *AllowedCall) {
  class NestedCallVisitor
      : public RecursiveASTVisitor<NestedCallVisitor> {
  public:
    NestedCallVisitor(Sema &S, const CallExpr *AllowedCall)
        : S(S), AllowedCall(AllowedCall) {}

    bool VisitCallExpr(CallExpr *Call) {
      if (Call == AllowedCall)
        return true;
      if (const FunctionDecl *Callee = Call->getDirectCallee()) {
        if (S.GTaP().isGTaPTaskFunction(Callee)) {
          S.Diag(Call->getExprLoc(), diag::err_gtap_direct_task_function_call)
              << Callee;
          Invalid = true;
        }
      }
      return true;
    }

    bool isInvalid() const { return Invalid; }

  private:
    Sema &S;
    const CallExpr *AllowedCall;
    bool Invalid = false;
  } Visitor(S, AllowedCall);

  Visitor.TraverseStmt(Root);
  return Visitor.isInvalid();
}

static CallExpr *getGTaPAssociatedCall(Stmt *S) {
  if (auto *BO = dyn_cast_or_null<BinaryOperator>(S)) {
    if (BO->getOpcode() == BO_Assign)
      return dyn_cast<CallExpr>(BO->getRHS()->IgnoreParenImpCasts());
  }
  if (auto *E = dyn_cast_or_null<Expr>(S))
    return dyn_cast<CallExpr>(E->IgnoreParenImpCasts());
  return nullptr;
}

StmtResult SemaGTaP::ActOnGTaPTaskDirective(Stmt *AStmt, SourceLocation StartLoc,
                                          SourceLocation EndLoc, Expr *QueueExpr) {
  if (!AStmt)
    return StmtError();

  ASTContext &Ctx = getASTContext();
  Expr *Q = QueueExpr ? QueueExpr : defaultQueueExpr(Ctx, StartLoc);
  CallExpr *SpawnCall = getGTaPAssociatedCall(AStmt);
  if (diagnoseNestedGTaPTaskFunctionCalls(SemaRef, AStmt, SpawnCall))
    return StmtError();

  return GTaPTaskDirective::Create(getASTContext(), StartLoc, EndLoc, AStmt, Q);
}

StmtResult SemaGTaP::ActOnGTaPTaskwaitDirective(SourceLocation StartLoc,
                                              SourceLocation EndLoc,
                                              Expr *QueueExpr) {
  if (SemaRef.getLangOpts().GTaPNoTaskwait ||
      isMacroDefined(SemaRef, "GTAP_ASSUME_NO_TASKWAIT")) {
    SemaRef.Diag(StartLoc, diag::err_gtap_taskwait_with_no_taskwait);
    return StmtError();
  }

  ASTContext &Ctx = getASTContext();
  Expr *Q = QueueExpr ? QueueExpr : defaultQueueExpr(Ctx, StartLoc);

  return GTaPTaskwaitDirective::Create(getASTContext(), StartLoc, EndLoc, Q);
}

FunctionDecl* SemaGTaP::getOrCreateStateMachineFunction(FunctionDecl *UserFD,
                                                       QualType VoidTy,
                                                       QualType VoidPtrTy,
                                                       QualType IntTy,
                                                       QualType TaskCtxPtrTy) {
  UserFD = UserFD->getCanonicalDecl();
  auto &TI = CachedTaskInfos[UserFD];
  if (TI.StateMachineFD) return TI.StateMachineFD;

  ASTContext &Ctx = getASTContext();
  DeclContext *DC = Ctx.getTranslationUnitDecl();

  std::string Name = "__gtap_state_machine_" + UserFD->getName().str() + "_" +
                     std::to_string(reinterpret_cast<uintptr_t>(UserFD));

  IdentifierInfo &II = Ctx.Idents.get(Name);

  FunctionProtoType::ExtProtoInfo EPI;
  QualType StateMachineFnTy = Ctx.getFunctionType(VoidTy, {VoidPtrTy, IntTy, TaskCtxPtrTy}, EPI);

  FunctionDecl *StateMachineFD = FunctionDecl::Create(
      Ctx, DC, UserFD->getBeginLoc(), UserFD->getLocation(),
      &II, StateMachineFnTy, Ctx.getTrivialTypeSourceInfo(StateMachineFnTy),
      SC_None);
  // StateMachineFD->setImplicit();

  // copy CUDA attributes
  if (UserFD->hasAttr<CUDADeviceAttr>())
    StateMachineFD->addAttr(CUDADeviceAttr::CreateImplicit(Ctx));

  auto mkParam = [&](StringRef N, QualType QT) {
    IdentifierInfo &PI = Ctx.Idents.get(N);
    return ParmVarDecl::Create(Ctx, StateMachineFD, SourceLocation(), SourceLocation(),
                               &PI, QT, Ctx.getTrivialTypeSourceInfo(QT),
                               SC_None, nullptr);
  };
  ParmVarDecl *SelfP = mkParam("__gtap_self",     VoidPtrTy);
  ParmVarDecl *TidP  = mkParam("__gtap_self_tid", IntTy);
  ParmVarDecl *CtxP  = mkParam("__gtap_ctx",      TaskCtxPtrTy);
  StateMachineFD->setParams({SelfP, TidP, CtxP});

  DC->addDecl(StateMachineFD);
  return (TI.StateMachineFD = StateMachineFD);
}

StmtResult SemaGTaP::ActOnGTaPInitDirective(SourceLocation StartLoc,
                                          SourceLocation EndLoc,
                                          StringRef RT, StringRef FN) {
  ASTContext &Ctx = getASTContext();

  FunctionDecl *InitFn = requireRuntimeFunction(SemaRef, "__gtap_init_task_runtime", StartLoc);
  if (!InitFn)
    return StmtError();

  // __gtap_init_task_runtime();
  ExprResult Callee = SemaRef.BuildDeclRefExpr(
    InitFn, InitFn->getType(), VK_LValue, StartLoc);

  ExprResult CallResult = SemaRef.BuildCallExpr(
    /*Scope=*/nullptr,
    Callee.get(),
    StartLoc,
    MultiExprArg(),
    EndLoc);

  if (CallResult.isInvalid())
    return StmtResult(new (Ctx) NullStmt(StartLoc));

  return SemaRef.ActOnExprStmt(CallResult.get());
}

StmtResult SemaGTaP::ActOnGTaPEntryDirective(SourceLocation StartLoc,
                                           SourceLocation EndLoc,
                                           Stmt *AStmt) {
  // Transform #pragma gtap entry - entry is only allowed inside __global__ kernel
  if (!AStmt) return StmtError();
  ASTContext &Ctx = getASTContext();
  
  // Extract call expression and result variable
  // Expected patterns:
  //   result = fib(n);  (assignment with call)
  //   fib(n);           (direct call)
  CallExpr *EntryCall = nullptr;
  VarDecl *ResultVar = nullptr;
  SmallVector<Expr *, 4> CallArgs;
  
  if (auto *BinOp = dyn_cast<BinaryOperator>(AStmt)) {
    if (BinOp->getOpcode() == BO_Assign) {
      // Get result variable from LHS
      if (auto *DRE = dyn_cast<DeclRefExpr>(BinOp->getLHS()->IgnoreParenImpCasts())) {
        ResultVar = dyn_cast<VarDecl>(DRE->getDecl());
      }
      // Get call expression from RHS
      Expr *RHS = BinOp->getRHS()->IgnoreParenImpCasts();
      EntryCall = dyn_cast<CallExpr>(RHS);
    }
  } else {
    EntryCall = dyn_cast<CallExpr>(AStmt);
  }
  
  if (!EntryCall) {
    SemaRef.Diag(StartLoc, diag::err_gtap_entry_invalid_statement);
    return StmtError();
  }

  if (diagnoseNestedGTaPTaskFunctionCalls(SemaRef, AStmt, EntryCall))
    return StmtError();
  
  FunctionDecl *CalleeDecl = EntryCall->getDirectCallee();
  if (!CalleeDecl) {
    SemaRef.Diag(StartLoc, diag::err_gtap_entry_no_callee);
    return StmtError();
  }
  
  // Extract arguments
  for (unsigned I = 0; I < EntryCall->getNumArgs(); ++I) {
    CallArgs.push_back(EntryCall->getArg(I));
  }
  
  std::string FuncName = CalleeDecl->getNameAsString();
  // llvm::errs() << "[GTaP][Sema] Entry function: " << FuncName << "\n";
  
  // Get task info for the callee
  GTaPTaskFunctionInfo &TaskInfo = getCachedTaskInfo(CalleeDecl);
  
  // Get task record
  llvm::DenseMap<const ValueDecl *, FieldDecl *> FieldMap;
  RecordDecl *TaskRecord = createTaskDataRecord(SemaRef, CalleeDecl, TaskInfo, FieldMap);
  if (!TaskRecord) {
    if (TaskInfo.TaskRecordInvalid)
      return StmtError();
    SemaRef.Diag(StartLoc, diag::err_gtap_entry_function_not_initialized) << FuncName;
    return StmtError();
  }
  
  QualType TaskRecordTy = Ctx.getTypeDeclType(cast<TypeDecl>(TaskRecord));
  if (SemaRef.RequireCompleteType(StartLoc, TaskRecordTy, diag::err_typecheck_incomplete_tag)) {
    return StmtError();
  }
  
  // llvm::errs() << "[GTaP][Sema] TaskRecord type: " << TaskRecordTy.getAsString() << "\n";
  
  // llvm::errs() << "[GTaP][Sema] Generating device entry code for function '"
  //              << FuncName << "'\n";
  
  SmallVector<Stmt *, 16> DeviceStmts;
  
  // Create helper function to build task pointer declaration and initialization
  QualType IntTy = Ctx.IntTy;
  QualType TaskPtrTy = Ctx.getPointerType(TaskRecordTy);
  const bool IsBlockWorker =
      isMacroDefined(SemaRef, "__GTAP_WORKER_IS_BLOCK");
  std::string TaskPtrName = "__gtap_task_ptr_" + FuncName;
  
  FunctionDecl *GetTaskDataFn = requireRuntimeFunction(SemaRef, "__gtap_get_task_data", StartLoc);
  if (!GetTaskDataFn)
    return StmtError();
  
  auto createTaskPtrDecl = [&]() -> std::pair<VarDecl*, Stmt*> {
    // Call __gtap_get_task_data(0)
    SmallVector<Expr *, 1> GetTaskDataArgs;
    GetTaskDataArgs.push_back(IntegerLiteral::Create(
        Ctx, llvm::APInt(Ctx.getIntWidth(IntTy), 0), IntTy, StartLoc));
    
    ExprResult GetTaskDataCallee = SemaRef.BuildDeclRefExpr(
        GetTaskDataFn, GetTaskDataFn->getType(), VK_LValue, StartLoc);
    ExprResult GetTaskDataCall = SemaRef.BuildCallExpr(
        nullptr, GetTaskDataCallee.get(), StartLoc, GetTaskDataArgs, EndLoc);
    
    if (GetTaskDataCall.isInvalid()) return {nullptr, nullptr};
    
    // Cast __gtap_get_task_data(0) to fib_task_data*
    Expr *CastExpr = CStyleCastExpr::Create(
        Ctx, TaskPtrTy, VK_PRValue, CK_BitCast,
        GetTaskDataCall.get(), nullptr,
        FPOptionsOverride(),
        Ctx.getTrivialTypeSourceInfo(TaskPtrTy),
        StartLoc, EndLoc);
    
    // Declare: fib_task_data *__gtap_task_ptr_fib = (fib_task_data*)__gtap_get_task_data(0);
    IdentifierInfo &TaskPtrId = Ctx.Idents.get(TaskPtrName);
    VarDecl *NewTaskPtrVar = VarDecl::Create(
        Ctx, SemaRef.CurContext, StartLoc, StartLoc,
        &TaskPtrId, TaskPtrTy, Ctx.getTrivialTypeSourceInfo(TaskPtrTy),
        SC_None);
    NewTaskPtrVar->setInit(CastExpr);
    
    DeclStmt *DeclS = new (Ctx) DeclStmt(DeclGroupRef(NewTaskPtrVar), StartLoc, EndLoc);
    return {NewTaskPtrVar, DeclS};
  };
  
  // Create task pointer and initialize fields
  auto [TaskPtrVar, TaskPtrDeclStmt] = createTaskPtrDecl();

  GTaPExprBuilder B(SemaRef, TaskPtrVar);
  
  if (TaskPtrVar && TaskPtrDeclStmt) {
    DeviceStmts.push_back(TaskPtrDeclStmt);
    // llvm::errs() << "[GTaP][Sema] Created task pointer (local to init block): " << TaskPtrName << "\n";
    
    // Initialize task fields with function arguments using pointer
    unsigned ArgIndex = 0;
    for (Expr *Arg : CallArgs) {
      if (ArgIndex >= TaskInfo.ParameterFields.size())
        break;
      
      FieldDecl *Field = TaskInfo.ParameterFields[ArgIndex];
      if (!Field)
        continue;
      
      // Create pointer dereference: *__gtap_task_ptr_fib
      Expr *TaskPtrRef = DeclRefExpr::Create(
          Ctx, NestedNameSpecifierLoc(), SourceLocation(), TaskPtrVar,
          false, StartLoc, TaskPtrVar->getType(), VK_LValue);
      Expr *TaskPtrRValue = ImplicitCastExpr::Create(
          Ctx, TaskPtrVar->getType(), CK_LValueToRValue,
          TaskPtrRef, nullptr, VK_PRValue, FPOptionsOverride());
      
      // Create arrow member access: __gtap_task_ptr_fib->param
      ExprResult FieldAccessER = B.buildFieldAccess(TaskPtrRValue, true, Field, SourceLocation());
      if (FieldAccessER.isInvalid()) return StmtError();
      Expr *FieldAccess = FieldAccessER.get();

      const bool IsPerThreadBlockField =
          IsBlockWorker && Ctx.getAsConstantArrayType(Field->getType());
      if (!IsPerThreadBlockField) {
        ExprResult AssignER = SemaRef.BuildBinOp(
            nullptr, SourceLocation(), BO_Assign, FieldAccess, Arg);
        if (AssignER.isInvalid()) return StmtError();
        DeviceStmts.push_back(AssignER.get());
      } else {
        Expr *Zero = IntegerLiteral::Create(
            Ctx, llvm::APInt(Ctx.getIntWidth(IntTy), 0), IntTy, StartLoc);
        Expr *Indices[] = {Zero};
        ExprResult Element = SemaRef.ActOnArraySubscriptExpr(
            nullptr, FieldAccess, StartLoc, MultiExprArg(Indices, 1), EndLoc);
        if (Element.isInvalid()) return StmtError();
        ExprResult Assign = SemaRef.BuildBinOp(
            nullptr, SourceLocation(), BO_Assign, Element.get(), Arg);
        if (Assign.isInvalid()) return StmtError();
        DeviceStmts.push_back(Assign.get());
      }
      ++ArgIndex;
    }
    
    // The root task has no spawning parent thread. Preserve the existing
    // block-mode entry convention in which thread 0 supplies its result.
    if (TaskInfo.SpawningThreadField) {
      Expr *TaskPtrRef = DeclRefExpr::Create(
          Ctx, NestedNameSpecifierLoc(), SourceLocation(), TaskPtrVar,
          false, StartLoc, TaskPtrVar->getType(), VK_LValue);
      Expr *TaskPtrRValue = ImplicitCastExpr::Create(
          Ctx, TaskPtrVar->getType(), CK_LValueToRValue,
          TaskPtrRef, nullptr, VK_PRValue, FPOptionsOverride());
      ExprResult FieldER = B.buildFieldAccess(
          TaskPtrRValue, true, TaskInfo.SpawningThreadField,
          SourceLocation());
      if (FieldER.isInvalid())
        return StmtError();
      Expr *Zero = IntegerLiteral::Create(
          Ctx, llvm::APInt(Ctx.getIntWidth(IntTy), 0), IntTy, StartLoc);
      ExprResult AssignER = SemaRef.BuildBinOp(
          nullptr, SourceLocation(), BO_Assign, FieldER.get(), Zero);
      if (AssignER.isInvalid())
        return StmtError();
      DeviceStmts.push_back(AssignER.get());
    }

    if (TaskInfo.ResultDstField && TaskInfo.ResultField) {
      Expr *TaskPtrRef = DeclRefExpr::Create(
          Ctx, NestedNameSpecifierLoc(), SourceLocation(), TaskPtrVar,
          false, StartLoc, TaskPtrVar->getType(), VK_LValue);
      Expr *TaskPtrRValue = ImplicitCastExpr::Create(
          Ctx, TaskPtrVar->getType(), CK_LValueToRValue,
          TaskPtrRef, nullptr, VK_PRValue, FPOptionsOverride());
      ExprResult FieldER = B.buildFieldAccess(
          TaskPtrRValue, true, TaskInfo.ResultDstField, SourceLocation());
      ExprResult ResultER = B.buildFieldAccess(
          TaskPtrRValue, true, TaskInfo.ResultField, SourceLocation());
      if (FieldER.isInvalid() || ResultER.isInvalid())
        return StmtError();
      Expr *OwnResultAddr = UnaryOperator::Create(
          Ctx, ResultER.get(), UO_AddrOf,
          TaskInfo.ResultDstField->getType(), VK_PRValue,
          OK_Ordinary, StartLoc, false, FPOptionsOverride());
      ExprResult AssignER = SemaRef.BuildBinOp(
          nullptr, SourceLocation(), BO_Assign,
          FieldER.get(), OwnResultAddr);
      if (AssignER.isInvalid())
        return StmtError();
      DeviceStmts.push_back(AssignER.get());
    }

  }

  // Initialize function pointer directly from CalleeDecl
  QualType VoidTy = Ctx.VoidTy;
  QualType VoidPtrTy = Ctx.getPointerType(VoidTy);
  QualType TaskCtxPtrTy = VoidPtrTy;
  if (QualType TaskCtxTy = lookupNamedType(SemaRef, "TaskContext"); !TaskCtxTy.isNull())
    TaskCtxPtrTy = Ctx.getPointerType(TaskCtxTy);

  SmallVector<QualType, 3> FuncPtrParamTypes = {VoidPtrTy, IntTy, TaskCtxPtrTy};
  FunctionProtoType::ExtProtoInfo FuncPtrEPI;
  QualType FuncPtrFuncType = Ctx.getFunctionType(VoidTy, FuncPtrParamTypes, FuncPtrEPI);
  QualType FuncPtrType = Ctx.getPointerType(FuncPtrFuncType);

  // Declare: void (*__gtap_func_ptr)(void*, int, TaskContext*)
  IdentifierInfo &FuncPtrId = Ctx.Idents.get("__gtap_func_ptr");
  VarDecl *FuncPtrVar = VarDecl::Create(
      Ctx, SemaRef.CurContext, StartLoc, StartLoc,
      &FuncPtrId, FuncPtrType, Ctx.getTrivialTypeSourceInfo(FuncPtrType),
      SC_None);

  FunctionDecl *StateMachineFD =
      CachedTaskInfos[CalleeDecl->getCanonicalDecl()].StateMachineFD;
  if (!StateMachineFD) StateMachineFD = getOrCreateStateMachineFunction(CalleeDecl, VoidTy, VoidPtrTy, IntTy, TaskCtxPtrTy);
  ExprResult CalleeRefER = SemaRef.BuildDeclRefExpr(StateMachineFD, StateMachineFD->getType(), VK_LValue, StartLoc);
  if (CalleeRefER.isInvalid()) return StmtError();
  Expr *CalleeRef = CalleeRefER.get();

  // Decay function to pointer if needed
  Expr *CalleeAsPtr = ImplicitCastExpr::Create(
      Ctx,
      Ctx.getPointerType(StateMachineFD->getType()),
      CK_FunctionToPointerDecay,
      CalleeRef,
      nullptr,
      VK_PRValue,
      FPOptionsOverride());

  // Cast to expected runtime signature (bitcast)
  Expr *FuncPtrInit = CStyleCastExpr::Create(
      Ctx, FuncPtrType, VK_PRValue, CK_BitCast,
      CalleeAsPtr, nullptr,
      FPOptionsOverride(),
      Ctx.getTrivialTypeSourceInfo(FuncPtrType),
      StartLoc, EndLoc);

  FuncPtrVar->setInit(FuncPtrInit);
  DeviceStmts.push_back(new (Ctx) DeclStmt(DeclGroupRef(FuncPtrVar), StartLoc, EndLoc));

  // llvm::errs() << "[GTaP][Sema] Initialized __gtap_func_ptr from CalleeDecl directly (no function table)\n";
  
  // Call __gtap_push_initial_task(__gtap_func_ptr, 0)
  FunctionDecl *PushTaskDeviceFn = requireRuntimeFunction(SemaRef, "__gtap_push_initial_task", StartLoc);
  if (!PushTaskDeviceFn)
    return StmtError();
  
  SmallVector<Expr *, 2> PushArgs;
  
  // Argument 1: func_ptr
  Expr *FuncPtrRef = DeclRefExpr::Create(
      Ctx, NestedNameSpecifierLoc(), SourceLocation(), FuncPtrVar,
      false, StartLoc, FuncPtrVar->getType(), VK_LValue);
  PushArgs.push_back(ImplicitCastExpr::Create(
      Ctx, FuncPtrVar->getType(), CK_LValueToRValue, FuncPtrRef, nullptr,
      VK_PRValue, FPOptionsOverride()));
  
  // Argument 2: task_kind = 0
  PushArgs.push_back(IntegerLiteral::Create(
      Ctx, llvm::APInt(Ctx.getIntWidth(IntTy), 0), IntTy, StartLoc));
  
  // Create call
  ExprResult PushCallee = SemaRef.BuildDeclRefExpr(
      PushTaskDeviceFn, PushTaskDeviceFn->getType(), VK_LValue, StartLoc);
  ExprResult PushCall = SemaRef.BuildCallExpr(
      nullptr, PushCallee.get(), StartLoc, PushArgs, EndLoc);
  
  if (!PushCall.isInvalid()) {
    DeviceStmts.push_back(PushCall.get());
    // llvm::errs() << "[GTaP][Sema] Generated __gtap_push_initial_task call\n";
  }
  
  // llvm::errs() << "[GTaP][Sema] Finished initial task setup code\n";
  
  // Execute task loop on device (ALL THREADS)
  Stmt *ExecuteStmt = nullptr;
  FunctionDecl *ExecuteLoopFn = requireRuntimeFunction(SemaRef, "__gtap_execute_task_loop_device", StartLoc);
  if (!ExecuteLoopFn)
    return StmtError();
  
  SmallVector<Expr *, 0> ExecuteArgs;
  ExprResult ExecuteCallee = SemaRef.BuildDeclRefExpr(
      ExecuteLoopFn, ExecuteLoopFn->getType(), VK_LValue, StartLoc);
  ExprResult ExecuteCall = SemaRef.BuildCallExpr(
      nullptr, ExecuteCallee.get(), StartLoc, ExecuteArgs, EndLoc);
  
  if (!ExecuteCall.isInvalid()) {
    ExecuteStmt = ExecuteCall.get();
    // llvm::errs() << "[GTaP][Sema] Generated __gtap_execute_task_loop_device call (for all threads)\n";
  }
  
  // Access result from task pointer (THREAD 0 ONLY)
  SmallVector<Stmt *, 8> ResultStmts;
  
  if (ResultVar && TaskInfo.ResultField && GetTaskDataFn) {
    // Create a new local task pointer declaration
    auto [ResultTaskPtrVar, ResultTaskPtrDeclStmt] = createTaskPtrDecl();
    
    if (ResultTaskPtrVar && ResultTaskPtrDeclStmt) {
      ResultStmts.push_back(ResultTaskPtrDeclStmt);
      
      // Access: __gtap_task_ptr_fib->__gtap_result
      Expr *TaskPtrRef = DeclRefExpr::Create(
          Ctx, NestedNameSpecifierLoc(), SourceLocation(), ResultTaskPtrVar,
          false, StartLoc, ResultTaskPtrVar->getType(), VK_LValue);
      Expr *TaskPtrRValue = ImplicitCastExpr::Create(
          Ctx, ResultTaskPtrVar->getType(), CK_LValueToRValue,
          TaskPtrRef, nullptr, VK_PRValue, FPOptionsOverride());
      
      // Create arrow member access: __gtap_task_ptr_fib->__gtap_result
      ExprResult ResultAccessER = B.buildFieldAccess(TaskPtrRValue, true, TaskInfo.ResultField, SourceLocation());
      if (ResultAccessER.isInvalid()) return StmtError();
      Expr *ResultAccess = ResultAccessER.get();
      
      // Convert to rvalue if needed
      if (ResultAccess->isLValue()) {
        ResultAccess = ImplicitCastExpr::Create(
            Ctx, ResultAccess->getType(), CK_LValueToRValue,
            ResultAccess, nullptr, VK_PRValue, FPOptionsOverride());
      }
      
      // Create assignment: d_result = __gtap_task_ptr_fib->__gtap_result
      Expr *ResultVarRef = DeclRefExpr::Create(
          Ctx, NestedNameSpecifierLoc(), SourceLocation(), ResultVar,
          false, StartLoc, ResultVar->getType(), VK_LValue);
      
      ExprResult ResultAssignER = SemaRef.BuildBinOp(nullptr, SourceLocation(), BO_Assign, ResultVarRef, ResultAccess);
      if (ResultAssignER.isInvalid()) return StmtError();
      Expr *ResultAssign = ResultAssignER.get();
      
      ResultStmts.push_back(ResultAssign);
      // llvm::errs() << "[GTaP][Sema] Generated direct result access with local pointer: "
      //              << "d_result = __gtap_task_ptr->__gtap_result\n";
    }
  }
  
  // llvm::errs() << "[GTaP][Sema] Device entry code generation complete\n";
  
  // Build final statement list:
  // 1. bool __gtap_is_master = (blockIdx.x == 0 && threadIdx.x == 0);
  // 2. if (__gtap_is_master) { InitStmts }
  // 3. ExecuteStmt (all threads)
  // 4. if (__gtap_is_master) { ResultStmts }
  
  SmallVector<Stmt *, 16> FinalStmts;
  
  // Build condition: blockIdx.x == 0 && threadIdx.x == 0
  auto buildThreadCondition = [&]() -> Expr* {
    // Lookup blockIdx and threadIdx CUDA built-in variables
    IdentifierInfo &BlockIdxId = Ctx.Idents.get("blockIdx");
    IdentifierInfo &ThreadIdxId = Ctx.Idents.get("threadIdx");
    
    DeclContext::lookup_result BlockIdxLookup = 
        Ctx.getTranslationUnitDecl()->lookup(&BlockIdxId);
    DeclContext::lookup_result ThreadIdxLookup = 
        Ctx.getTranslationUnitDecl()->lookup(&ThreadIdxId);
    
    VarDecl *BlockIdxVar = nullptr;
    VarDecl *ThreadIdxVar = nullptr;
    
    if (!BlockIdxLookup.empty()) {
      BlockIdxVar = dyn_cast<VarDecl>(BlockIdxLookup.front());
    }
    if (!ThreadIdxLookup.empty()) {
      ThreadIdxVar = dyn_cast<VarDecl>(ThreadIdxLookup.front());
    }
    
    if (!BlockIdxVar || !ThreadIdxVar)
      return nullptr;
    
    // Build blockIdx.x
    Expr *BlockIdxRef = DeclRefExpr::Create(
        Ctx, NestedNameSpecifierLoc(), SourceLocation(), BlockIdxVar,
        false, StartLoc, BlockIdxVar->getType(), VK_LValue);
    
    CXXScopeSpec SS;
    IdentifierInfo &XId = Ctx.Idents.get("x");
    DeclarationNameInfo XNameInfo(&XId, StartLoc);
    
    ExprResult BlockIdxXResult = SemaRef.BuildMemberReferenceExpr(
        BlockIdxRef, BlockIdxVar->getType(), StartLoc, false, SS,
        SourceLocation(), nullptr, XNameInfo, nullptr, nullptr);
    
    if (BlockIdxXResult.isInvalid()) return nullptr;
    
    Expr *Zero1 = IntegerLiteral::Create(
        Ctx, llvm::APInt(Ctx.getIntWidth(Ctx.IntTy), 0), Ctx.IntTy, StartLoc);
    
    ExprResult BlockCond = SemaRef.BuildBinOp(
        nullptr, StartLoc, BO_EQ, BlockIdxXResult.get(), Zero1);
    
    if (BlockCond.isInvalid()) return nullptr;
    
    // Build threadIdx.x
    Expr *ThreadIdxRef = DeclRefExpr::Create(
        Ctx, NestedNameSpecifierLoc(), SourceLocation(), ThreadIdxVar,
        false, StartLoc, ThreadIdxVar->getType(), VK_LValue);
    
    ExprResult ThreadIdxXResult = SemaRef.BuildMemberReferenceExpr(
        ThreadIdxRef, ThreadIdxVar->getType(), StartLoc, false, SS,
        SourceLocation(), nullptr, XNameInfo, nullptr, nullptr);
    
    if (ThreadIdxXResult.isInvalid()) return nullptr;
    
    Expr *Zero2 = IntegerLiteral::Create(
        Ctx, llvm::APInt(Ctx.getIntWidth(Ctx.IntTy), 0), Ctx.IntTy, StartLoc);
    
    ExprResult ThreadCond = SemaRef.BuildBinOp(
        nullptr, StartLoc, BO_EQ, ThreadIdxXResult.get(), Zero2);
    
    if (ThreadCond.isInvalid()) return nullptr;
    
    // Combine with &&
    ExprResult Combined = SemaRef.BuildBinOp(
        nullptr, StartLoc, BO_LAnd, BlockCond.get(), ThreadCond.get());
    
    return Combined.isInvalid() ? nullptr : Combined.get();
  };
  
  // Create bool __gtap_is_master = (blockIdx.x == 0 && threadIdx.x == 0);
  Expr *MasterCond = buildThreadCondition();
  if (MasterCond) {
      IdentifierInfo &MasterVarId = Ctx.Idents.get("__gtap_is_master");
      VarDecl *MasterVar = VarDecl::Create(
          Ctx, SemaRef.CurContext, StartLoc, StartLoc,
          &MasterVarId, Ctx.BoolTy, Ctx.getTrivialTypeSourceInfo(Ctx.BoolTy),
          SC_None);
      MasterVar->setInit(MasterCond);
      
      DeclStmt *MasterVarDecl = new (Ctx) DeclStmt(
          DeclGroupRef(MasterVar), StartLoc, EndLoc);
      FinalStmts.push_back(MasterVarDecl);
      // llvm::errs() << "[GTaP][Sema] Created __gtap_is_master variable\n";
      
      // DeviceStmts contains all init statements including task pointer declaration
      SmallVector<Stmt *, 16> InitStmts(DeviceStmts.begin(), DeviceStmts.end());
      
      // Create first if statement for initialization
      // if (__gtap_is_master) { InitStmts }
      Expr *MasterVarRef1 = DeclRefExpr::Create(
          Ctx, NestedNameSpecifierLoc(), SourceLocation(), MasterVar,
          false, StartLoc, Ctx.BoolTy, VK_LValue);
      Expr *MasterCheck1 = ImplicitCastExpr::Create(
          Ctx, Ctx.BoolTy, CK_LValueToRValue, MasterVarRef1, nullptr,
          VK_PRValue, FPOptionsOverride());
      
      CompoundStmt *InitBlock = CompoundStmt::Create(
          Ctx, InitStmts, FPOptionsOverride(), StartLoc, EndLoc);
      
      IfStmt *InitIfStmt = IfStmt::Create(
          Ctx, StartLoc,
          IfStatementKind::Ordinary,
          nullptr,  // init
          nullptr,  // conditionVariable
          MasterCheck1,  // cond
          StartLoc,  // LParenLoc
          StartLoc,  // RParenLoc
          InitBlock,  // then
          SourceLocation(),  // elseLoc
          nullptr  // elseStmt
      );
      
      FinalStmts.push_back(InitIfStmt);
      // llvm::errs() << "[GTaP][Sema] Created first if (__gtap_is_master) block for initialization\n";
      
      // Add execute statement (all threads)
      if (ExecuteStmt) {
        FinalStmts.push_back(ExecuteStmt);
        // llvm::errs() << "[GTaP][Sema] Added execute_task_loop_device call (runs on all threads)\n";
      }
      
      // Create second if statement for result
      if (!ResultStmts.empty()) {
        Expr *MasterVarRef2 = DeclRefExpr::Create(
            Ctx, NestedNameSpecifierLoc(), SourceLocation(), MasterVar,
            false, StartLoc, Ctx.BoolTy, VK_LValue);
        Expr *MasterCheck2 = ImplicitCastExpr::Create(
            Ctx, Ctx.BoolTy, CK_LValueToRValue, MasterVarRef2, nullptr,
            VK_PRValue, FPOptionsOverride());
        
        CompoundStmt *ResultBlock = CompoundStmt::Create(
            Ctx, ResultStmts, FPOptionsOverride(), StartLoc, EndLoc);
        
        IfStmt *ResultIfStmt = IfStmt::Create(
            Ctx, StartLoc,
            IfStatementKind::Ordinary,
            nullptr,  // init
            nullptr,  // conditionVariable
            MasterCheck2,  // cond
            StartLoc,  // LParenLoc
            StartLoc,  // RParenLoc
            ResultBlock,  // then
            SourceLocation(),  // elseLoc
            nullptr  // elseStmt
        );
        
        FinalStmts.push_back(ResultIfStmt);
        // llvm::errs() << "[GTaP][Sema] Created second if (__gtap_is_master) block for result copy\n";
      }
      
      CompoundStmt *FinalBlock = CompoundStmt::Create(
          Ctx, FinalStmts, FPOptionsOverride(), StartLoc, EndLoc);
      
      // llvm::errs() << "[GTaP][Sema] Wrapped device code: bool __gtap_is_master, task decl, if(__gtap_is_master) init, execute all threads, if(__gtap_is_master) result\n";
      return StmtResult(FinalBlock);
  }
  
  // Fallback if condition building failed (blockIdx/threadIdx not found)
  FinalStmts.append(DeviceStmts.begin(), DeviceStmts.end());
  if (ExecuteStmt) FinalStmts.push_back(ExecuteStmt);
  FinalStmts.append(ResultStmts.begin(), ResultStmts.end());
  CompoundStmt *FallbackBlock = CompoundStmt::Create(
      Ctx, FinalStmts, FPOptionsOverride(), StartLoc, EndLoc);
  return StmtResult(FallbackBlock);
}

void SemaGTaP::ActOnFunctionDeclaration(FunctionDecl *FD) {
  if (!FD) return;
    
  // Check if there is a pending GTaP function pragma
  if (hasPendingGTaPFunctionPragma()) {
    // Attach GTaPFunctionAttr to this function
    FD->addAttr(GTaPFunctionAttr::Create(getASTContext(),
                                         PendingFunctionPragmaLoc));
    
    // Clear the pending pragma
    clearPendingGTaPFunctionPragma();
  }
}

void SemaGTaP::ActOnStartOfFunctionDef(FunctionDecl *FD) {
  ActOnFunctionDeclaration(FD);

  // A pragma attached to an earlier declaration may not have participated in
  // Clang's normal attribute merge because it is installed after that
  // declaration has been built.  Materialize it on the definition so the
  // definition is transformed like an inline-annotated task function.
  if (FD && !FD->hasAttr<GTaPFunctionAttr>()) {
    for (FunctionDecl *Redecl : FD->redecls()) {
      if (auto *A = Redecl->getAttr<GTaPFunctionAttr>()) {
        FD->addAttr(A->clone(getASTContext()));
        break;
      }
    }
  }
}

StmtResult SemaGTaP::TransformTaskFunctionBody(FunctionDecl *FD,
                                              CompoundStmt *Body) {
  if (!FD || !Body)
    return StmtError();

  ASTContext &Ctx = getASTContext();
  if (!checkNoTaskwaitInSwitchOrDiag(SemaRef, Body))
    return StmtError();
  // llvm::errs() << "[GTaP][Sema] Enter TransformTaskFunctionBody for function '"
  //              << FD->getName() << "'\n";

  auto asRValue = [&](Expr *E) -> Expr* {
    if (!E || !E->isGLValue()) return E;
    return ImplicitCastExpr::Create(Ctx, E->getType(), CK_LValueToRValue,
                                    E, nullptr, VK_PRValue, FPOptionsOverride());
  };

  GTaPTaskFunctionAnalyzer Analyzer(Ctx, FD);
  GTaPTaskFunctionInfo TaskInfo = Analyzer.analyze(Body);
  FunctionDecl *CacheKey = FD->getCanonicalDecl();
  CachedTaskInfos[CacheKey] = TaskInfo;

  // llvm::errs() << "[GTaP][Sema]  TaskInfo: params=" << TaskInfo.Parameters.size()
  //              << " captured=" << TaskInfo.CapturedVariables.size()
  //              << " directives=" << TaskInfo.Directives.size() << "\n";

  llvm::DenseMap<const ValueDecl *, FieldDecl *> FieldMap;
  RecordDecl *TaskRecord = createTaskDataRecord(
      SemaRef, FD, CachedTaskInfos[CacheKey], FieldMap);
  if (!TaskRecord)
    return StmtError();

  // llvm::errs() << "[GTaP][Sema]  Created/updated task record '"
  //              << TaskRecord->getName() << "'\n";

  QualType VoidTy = Ctx.VoidTy;
  QualType VoidPtrTy = Ctx.getPointerType(Ctx.VoidTy);
  QualType IntTy = Ctx.IntTy;
  const bool IsBlockWorker = isMacroDefined(SemaRef, "__GTAP_WORKER_IS_BLOCK");

  QualType TaskCtxPtrTy = Ctx.getPointerType(Ctx.VoidTy);
  QualType TaskCtxTy = lookupNamedType(SemaRef, "TaskContext");
  if (!TaskCtxTy.isNull())
    TaskCtxPtrTy = Ctx.getPointerType(TaskCtxTy);
  
  FunctionDecl *StateMachineFD = 
      getOrCreateStateMachineFunction(FD, VoidTy, VoidPtrTy, IntTy, TaskCtxPtrTy);
  ParmVarDecl *SelfParam = StateMachineFD->getParamDecl(0);
  ParmVarDecl *TidParam  = StateMachineFD->getParamDecl(1);
  ParmVarDecl *CtxParam  = StateMachineFD->getParamDecl(2);

  // self_typed
  QualType TaskRecordTy = Ctx.getTypeDeclType(static_cast<const TypeDecl *>(TaskRecord));
  QualType TaskPtrTy = Ctx.getPointerType(TaskRecordTy);

  Expr *SelfVoidLV = DeclRefExpr::Create(
    Ctx, NestedNameSpecifierLoc(), SourceLocation(),
    SelfParam, false, SourceLocation(), SelfParam->getType(), VK_LValue);
  Expr *SelfVoidRV = asRValue(SelfVoidLV);

  TypeSourceInfo *SelfTSI = Ctx.getTrivialTypeSourceInfo(TaskPtrTy);
  ExprResult CastER = SemaRef.BuildCXXNamedCast(
    SourceLocation(), tok::kw_reinterpret_cast, SelfTSI,
    SelfVoidRV, SourceLocation(), SourceLocation()
  );
  if (CastER.isInvalid()) return StmtError();
  Expr *SelfTypedInit = CastER.get();

  IdentifierInfo &SelfTypedId = Ctx.Idents.get("__gtap_self_typed");
  VarDecl *SelfTypedVar = VarDecl::Create(
    Ctx, StateMachineFD, SourceLocation(), SourceLocation(),
    &SelfTypedId, TaskPtrTy, Ctx.getTrivialTypeSourceInfo(TaskPtrTy), SC_None);
  SelfTypedVar->setInit(SelfTypedInit);
  DeclStmt *SelfTypedDeclStmt = 
    new (Ctx) DeclStmt(DeclGroupRef(SelfTypedVar), SourceLocation(), SourceLocation());

  VarDecl *ChildCountVar = nullptr;
  DeclStmt *ChildCountDeclStmt = nullptr;
  if (!IsBlockWorker) {
    IdentifierInfo &ChildCountId = Ctx.Idents.get("__gtap_child_count");
    ChildCountVar = VarDecl::Create(
        Ctx, StateMachineFD, SourceLocation(), SourceLocation(), &ChildCountId,
        IntTy, Ctx.getTrivialTypeSourceInfo(IntTy), SC_None);
    Expr *ChildCountZero = IntegerLiteral::Create(
        Ctx, llvm::APInt(Ctx.getIntWidth(IntTy), 0), IntTy, SourceLocation());
    ChildCountVar->setInit(ChildCountZero);
    ChildCountDeclStmt =
        new (Ctx) DeclStmt(DeclGroupRef(ChildCountVar), SourceLocation(),
                           SourceLocation());
  }

  // finish fn
  FunctionDecl *FinishFn = requireRuntimeFunction(SemaRef, "__gtap_finish_task", Body->getBeginLoc());
  if (!FinishFn)
    return StmtError();

  // transform body
  GTaPExprBuilder B(SemaRef, SelfTypedVar);
  GTaPTaskBodyTransformer Transformer(
      SemaRef, B, SelfTypedVar, FieldMap, CachedTaskInfos[CacheKey].ResultField,
      CachedTaskInfos[CacheKey].ResultDstField,
      CachedTaskInfos[CacheKey].SpawningThreadField,
      TidParam, CtxParam, FinishFn, ChildCountVar, IsBlockWorker);
  StmtResult Transformed = Transformer.TransformStmt(Body);
  if (Transformed.isInvalid())
    return StmtError();

  Stmt *TransformedStmt = Transformed.get();
  CompoundStmt *LinearizedBody = dyn_cast<CompoundStmt>(TransformedStmt);
  if (!LinearizedBody) {
    SmallVector<Stmt *, 4> Wrap = {TransformedStmt};
    LinearizedBody = CompoundStmt::Create(Ctx, Wrap, FPOptionsOverride(),
                                          SourceLocation(), SourceLocation());
  }

  // SwitchCond: __gtap_get_task_state(tid)
  auto buildParamLValue = [&](ParmVarDecl *P) -> Expr* {
    return DeclRefExpr::Create(Ctx, NestedNameSpecifierLoc(), SourceLocation(),
                               P, false, SourceLocation(), P->getType(), VK_LValue);
  };

  auto buildGetStateFromHeader = [&]() -> Expr * {
    FunctionDecl *GetStateFn = requireRuntimeFunction(SemaRef, "__gtap_get_task_state", Body->getBeginLoc());
    if (!GetStateFn)
      return nullptr;
    
    SmallVector<Expr *, 1> Args = {asRValue(buildParamLValue(TidParam))};
    ExprResult Callee = SemaRef.BuildDeclRefExpr(
        GetStateFn, GetStateFn->getType(), VK_LValue, SourceLocation());
    if (Callee.isInvalid())
      return nullptr;
    ExprResult Call = SemaRef.BuildCallExpr(
        /*Scope=*/nullptr, Callee.get(), SourceLocation(), Args,
        SourceLocation());
    if (Call.isInvalid())
      return nullptr;
    return Call.get();
  };

  Expr *SwitchCond = buildGetStateFromHeader();
  if (!SwitchCond) {
    // llvm::errs() << "[GTaP][Sema] Warning: State machine disabled (state access not implemented yet)\n";
    // llvm::errs() << "[GTaP][Sema] Generating linearized body as fallback\n";
    SmallVector<Stmt *, 3> Fallback;
    if (ChildCountDeclStmt)
      Fallback.push_back(ChildCountDeclStmt);
    Fallback.push_back(LinearizedBody);
    return StmtResult(CompoundStmt::Create(Ctx, Fallback, FPOptionsOverride(),
                                           Body->getLBracLoc(),
                                           Body->getRBracLoc()));
  }

  SwitchStmt *Switch =
      SwitchStmt::Create(Ctx, /*Init=*/nullptr, /*Var=*/nullptr, SwitchCond,
                         SourceLocation(), SourceLocation());

  // set_state_for_join call
  FunctionDecl *SetStateForJoinFn = requireRuntimeFunction(
      SemaRef,
      IsBlockWorker ? "__gtap_set_state_for_join_block" : "__gtap_set_state_for_join",
      Body->getBeginLoc());
  if (!SetStateForJoinFn)
    return StmtError();

  auto buildChildCountRValue = [&]() -> Expr* {
    if (!ChildCountVar)
      return IntegerLiteral::Create(
          Ctx, llvm::APInt(Ctx.getIntWidth(IntTy), 0), IntTy,
          SourceLocation());
    Expr *LV = DeclRefExpr::Create(
        Ctx, NestedNameSpecifierLoc(), SourceLocation(),
        ChildCountVar, /*RefersToEnclosingVariableOrCapture=*/false,
        SourceLocation(), ChildCountVar->getType(), VK_LValue);
    return asRValue(LV);
  };

  auto buildSetStateForJoinCall = [&](unsigned NextState, Expr *QueueExpr) -> Stmt* {
    if (!SetStateForJoinFn) return nullptr;

    auto buildWaitQueueArg = [&](Expr *QE) -> Expr* {
      if (!QE) return IntegerLiteral::Create(Ctx, llvm::APInt(Ctx.getIntWidth(Ctx.IntTy), 0), Ctx.IntTy, SourceLocation());
      Expr *E = asRValue(QE);
      if (!Ctx.hasSameType(E->getType(), IntTy)) {
        E = ImplicitCastExpr::Create(Ctx, IntTy, CK_IntegralCast, E,
                                     nullptr, VK_PRValue, FPOptionsOverride());
      }
      return E;
    };
    
    // args: (tid, child_count_or_ctx, next_state, queue)
    SmallVector<Expr*, 4> Args;
    Args.push_back(asRValue(buildParamLValue(TidParam)));
    Args.push_back(IsBlockWorker ? asRValue(buildParamLValue(CtxParam))
                                  : buildChildCountRValue());
    Args.push_back(IntegerLiteral::Create(
        Ctx, llvm::APInt(Ctx.getIntWidth(Ctx.IntTy), NextState),
        Ctx.IntTy, SourceLocation()));
    Args.push_back(buildWaitQueueArg(QueueExpr));
  
    ExprResult Callee = SemaRef.BuildDeclRefExpr(
        SetStateForJoinFn, SetStateForJoinFn->getType(), VK_LValue,
        SourceLocation());
    if (Callee.isInvalid())
      return nullptr;
    ExprResult Call = SemaRef.BuildCallExpr(
        /*Scope=*/nullptr, Callee.get(), SourceLocation(), Args,
        SourceLocation());
    if (Call.isInvalid())
      return nullptr;
    return Call.get();
  };

  auto appendSetStateForJoin = [&](SmallVectorImpl<Stmt *> &Out,
                                   unsigned NextState, Expr *QueueExpr) {
    Stmt *SetState = buildSetStateForJoinCall(NextState, QueueExpr);
    if (!SetState)
      return;

    Stmt *Return =
        ReturnStmt::Create(Ctx, SourceLocation(), nullptr, nullptr);
    Expr *Cond = dyn_cast<Expr>(SetState);
    if (!Cond) {
      Out.push_back(SetState);
      Out.push_back(Return);
      return;
    }
    Out.push_back(IfStmt::Create(
        Ctx, SourceLocation(), IfStatementKind::Ordinary,
        nullptr, nullptr, Cond, SourceLocation(), SourceLocation(),
        Return, SourceLocation(), nullptr));
  };

  // judge if the last statement is return
  auto endsWithReturnDeep = [&](auto&& self, Stmt *S) -> bool {
    if (!S) return false;
    if (auto *CS = dyn_cast<CompoundStmt>(S)) {
      for (auto it = CS->body_rbegin(); it != CS->body_rend(); ++it) {
        if (*it) return self(self, *it);
      }
      return false;
    }
    return isa<ReturnStmt>(S);
  };

  auto appendFinishAndReturnIfNeeded = [&](SmallVectorImpl<Stmt*> &stmts) {
    bool EndsWithReturn = !stmts.empty() && endsWithReturnDeep(endsWithReturnDeep, stmts.back());
    if (EndsWithReturn) return;

    if (FinishFn) {
      SmallVector<Expr*, 2> Args;
      Args.push_back(asRValue(buildParamLValue(TidParam)));
      Args.push_back(asRValue(buildParamLValue(CtxParam)));
      ExprResult Callee = SemaRef.BuildDeclRefExpr(FinishFn, FinishFn->getType(), VK_LValue, SourceLocation());
      if (!Callee.isInvalid()) {
        ExprResult Call = SemaRef.BuildCallExpr(nullptr, Callee.get(), SourceLocation(), Args, SourceLocation());
        if (!Call.isInvalid())
          stmts.push_back(Call.get());
      }
    }
    stmts.push_back(ReturnStmt::Create(Ctx, SourceLocation(), nullptr, nullptr));
  };

  // A block-task call is issued by one CUDA thread, but its ordinary function
  // parameters remain per-thread values.  The spawning thread initializes only
  // its own array element.  On the first invocation, every other worker thread
  // copies that element into its private slot.  This deliberately runs only in
  // state 0: repeating it after a taskwait would overwrite parameter mutations.
  auto buildBlockArgumentBroadcast = [&]() -> Stmt * {
    GTaPTaskFunctionInfo &TI = CachedTaskInfos[CacheKey];
    if (!IsBlockWorker || !TI.SpawningThreadField ||
        TI.ParameterFields.empty())
      return nullptr;

    SmallVector<Stmt *, 16> Copies;
    for (FieldDecl *Field : TI.ParameterFields) {
      if (!Field)
        continue;
      // Top-level const parameters use one uniform field for the whole block
      // and therefore require no state-0 lane broadcast.
      if (!Ctx.getAsConstantArrayType(Field->getType()))
        continue;

      ExprResult DstField = B.buildFieldAccess(
          B.buildSelfRef(), /*IsArrow=*/true, Field, SourceLocation());
      ExprResult SrcField = B.buildFieldAccess(
          B.buildSelfRef(), /*IsArrow=*/true, Field, SourceLocation());
      ExprResult SpawnLane = B.buildFieldAccess(
          B.buildSelfRef(), /*IsArrow=*/true, TI.SpawningThreadField,
          SourceLocation());
      Expr *DstLane = buildThreadIdxXExpr(SemaRef, SourceLocation());
      if (DstField.isInvalid() || SrcField.isInvalid() ||
          SpawnLane.isInvalid() || !DstLane)
        return nullptr;

      Expr *DstIndices[] = {DstLane};
      Expr *SrcIndices[] = {asRValue(SpawnLane.get())};
      ExprResult Dst = SemaRef.ActOnArraySubscriptExpr(
          nullptr, DstField.get(), SourceLocation(),
          MultiExprArg(DstIndices, 1), SourceLocation());
      ExprResult Src = SemaRef.ActOnArraySubscriptExpr(
          nullptr, SrcField.get(), SourceLocation(),
          MultiExprArg(SrcIndices, 1), SourceLocation());
      if (Dst.isInvalid() || Src.isInvalid())
        return nullptr;
      ExprResult Assign = SemaRef.BuildBinOp(
          nullptr, SourceLocation(), BO_Assign, Dst.get(), Src.get());
      if (Assign.isInvalid())
        return nullptr;
      Copies.push_back(Assign.get());
    }

    if (Copies.empty())
      return nullptr;

    Expr *ThreadIdxX = buildThreadIdxXExpr(SemaRef, SourceLocation());
    ExprResult SpawnLane = B.buildFieldAccess(
        B.buildSelfRef(), /*IsArrow=*/true, TI.SpawningThreadField,
        SourceLocation());
    if (!ThreadIdxX || SpawnLane.isInvalid())
      return nullptr;
    ExprResult IsNotSpawner = SemaRef.BuildBinOp(
        nullptr, SourceLocation(), BO_NE, ThreadIdxX,
        asRValue(SpawnLane.get()));
    if (IsNotSpawner.isInvalid())
      return nullptr;

    CompoundStmt *CopyBody = CompoundStmt::Create(
        Ctx, Copies, FPOptionsOverride(), SourceLocation(), SourceLocation());
    return IfStmt::Create(
        Ctx, SourceLocation(), IfStatementKind::Ordinary,
        nullptr, nullptr, IsNotSpawner.get(), SourceLocation(),
        SourceLocation(), CopyBody, SourceLocation(), nullptr);
  };


  // inject case at taskwait position
  SwitchCase *CaseList = nullptr;

  auto registerCase = [&](CaseStmt *CS) {
    CS->setNextSwitchCase(CaseList);
    CaseList = CS;
  };

  // ---- nested inject: only handles taskwait inside Compound/If/For/While/Do ----
  auto injectNestedTaskwaitCases = [&](auto&& self, Stmt *S) -> Stmt* {
    if (!S) return S;
    if (isa<SwitchStmt>(S)) return S; // nested switch is not allowed

    if (auto *CS = dyn_cast<CompoundStmt>(S)) {
      SmallVector<Stmt*, 32> bodyVec;
      bodyVec.append(CS->body_begin(), CS->body_end());

      SmallVector<Stmt*, 32> out;

      for (unsigned i = 0; i < bodyVec.size(); ++i) {
        Stmt *cur = bodyVec[i];

        if (auto *TW = dyn_cast<GTaPTaskwaitDirective>(cur)) {
          const unsigned waitId = TW->getWaitId();
          const unsigned resumeState = waitId + 1;

          SmallVector<Stmt*, 32> tailRaw;
          for (unsigned j = i + 1; j < bodyVec.size(); ++j)
            tailRaw.push_back(bodyVec[j]);

          SmallVector<Stmt*, 32> resumeBodyStmts;
          resumeBodyStmts.append(tailRaw.begin(), tailRaw.end());

          // resume body may contain another taskwait, so recurse inject here
          Stmt *ResumeBodyInjected = nullptr;
          {
            CompoundStmt *ResumeRawCS = CompoundStmt::Create(
                Ctx, resumeBodyStmts, FPOptionsOverride(),
                SourceLocation(), SourceLocation());
            ResumeBodyInjected = self(self, ResumeRawCS);
          }
          CompoundStmt *ResumeCS = dyn_cast<CompoundStmt>(ResumeBodyInjected);
          if (!ResumeCS) {
            SmallVector<Stmt*,1> Wrap = {ResumeBodyInjected};
            ResumeCS = CompoundStmt::Create(Ctx, Wrap, FPOptionsOverride(),
                                            SourceLocation(), SourceLocation());
          }

          // create resume case
          Expr *CaseValue = IntegerLiteral::Create(
              Ctx, llvm::APInt(Ctx.getIntWidth(IntTy), resumeState),
              IntTy, SourceLocation());

          CaseStmt *ResumeCase = CaseStmt::Create(
              Ctx, CaseValue, nullptr,
              SourceLocation(), SourceLocation(), SourceLocation());
          ResumeCase->setSubStmt(ResumeCS);

          // register to switch case-list
          registerCase(ResumeCase);

          // at taskwait position:
          appendSetStateForJoin(out, resumeState, TW->getQueueExpr());
          out.push_back(ResumeCase);

          return CompoundStmt::Create(Ctx, out, FPOptionsOverride(),
                                      CS->getLBracLoc(), CS->getRBracLoc());
        }
        out.push_back(self(self, cur));
      }

      return CompoundStmt::Create(Ctx, out, FPOptionsOverride(),
                                  CS->getLBracLoc(), CS->getRBracLoc());
    }

    if (auto *IF = dyn_cast<IfStmt>(S)) {
      IF->setThen(self(self, IF->getThen()));
      if (Stmt *E = IF->getElse())
        IF->setElse(self(self, E));
      return IF;
    }

    if (auto *FS = dyn_cast<ForStmt>(S)) {
      FS->setBody(self(self, FS->getBody()));
      return FS;
    }

    if (auto *WS = dyn_cast<WhileStmt>(S)) {
      WS->setBody(self(self, WS->getBody()));
      return WS;
    }

    if (auto *DS = dyn_cast<DoStmt>(S)) {
      DS->setBody(self(self, DS->getBody()));
      return DS;
    }

    return S;
  };

  // ---- top-level stage split ----
  struct TopStage {
    SmallVector<Stmt*, 32> Body;     // body without taskwait
    int ResumeWaitId = -1;           // taskwait id before entering this stage (stage0 is -1)
    Expr *ResumeQueueExpr = nullptr;
    int EndWaitId = -1;              // top-level taskwait id at the end of this stage (last stage is -1)
    Expr *EndQueueExpr = nullptr;
  };

  SmallVector<TopStage, 8> Stages;
  TopStage Cur;

  for (Stmt *Child : LinearizedBody->body()) {
    if (auto *TW = dyn_cast<GTaPTaskwaitDirective>(Child)) {
      // top-level taskwait: stage is determined
      Cur.EndWaitId = (int)TW->getWaitId();
      Cur.EndQueueExpr = TW->getQueueExpr();
      Stages.push_back(Cur);

      // next stage: resume starts from this taskwait
      TopStage Next;
      Next.ResumeWaitId = (int)TW->getWaitId();
      Next.ResumeQueueExpr = TW->getQueueExpr();
      Cur = Next;
      continue;
    }
    Cur.Body.push_back(Child);
  }
  Stages.push_back(Cur);

  // build top-level cases for each stage
  SmallVector<Stmt*, 16> TopLevelCaseStmts;

  for (unsigned si = 0; si < Stages.size(); ++si) {
    TopStage &St = Stages[si];

    // state value is "resumeWaitId + 1" (first stage is 0)
    const unsigned stateVal = (St.ResumeWaitId < 0) ? 0u : (unsigned)(St.ResumeWaitId + 1);

    SmallVector<Stmt*, 64> CaseStmts;

    if (stateVal == 0)
      if (Stmt *Broadcast = buildBlockArgumentBroadcast())
        CaseStmts.push_back(Broadcast);

    // Stage body: inject nested taskwait only here
    {
      CompoundStmt *StageRawCS = CompoundStmt::Create(
          Ctx, St.Body, FPOptionsOverride(),
          SourceLocation(), SourceLocation());

      Stmt *StageInjected = injectNestedTaskwaitCases(injectNestedTaskwaitCases, StageRawCS);
      CompoundStmt *StageCS = dyn_cast<CompoundStmt>(StageInjected);
      if (!StageCS) {
        SmallVector<Stmt*,1> Wrap = {StageInjected};
        StageCS = CompoundStmt::Create(Ctx, Wrap, FPOptionsOverride(),
                                       SourceLocation(), SourceLocation());
      }

      for (Stmt *SS : StageCS->body())
        CaseStmts.push_back(SS);
    }

    // (C) stage end: set_state + return if top-level taskwait exists
    if (St.EndWaitId >= 0) {
      const unsigned nextState = (unsigned)(St.EndWaitId + 1);
      appendSetStateForJoin(CaseStmts, nextState, St.EndQueueExpr);
    } else {
      // last stage: guarantee finish + return
      appendFinishAndReturnIfNeeded(CaseStmts);
    }

    CompoundStmt *CaseBody = CompoundStmt::Create(
        Ctx, CaseStmts, FPOptionsOverride(),
        SourceLocation(), SourceLocation());

    Expr *CaseValue = IntegerLiteral::Create(
        Ctx, llvm::APInt(Ctx.getIntWidth(IntTy), stateVal),
        IntTy, SourceLocation());

    CaseStmt *TopCase = CaseStmt::Create(
        Ctx, CaseValue, nullptr,
        SourceLocation(), SourceLocation(), SourceLocation());
    TopCase->setSubStmt(CaseBody);

    registerCase(TopCase);
    TopLevelCaseStmts.push_back(TopCase);
  }

  // ---- default: trap ----
  DefaultStmt *Default = nullptr;
  {
    FunctionDecl *TrapFn = nullptr;
    IdentifierInfo &TrapII = Ctx.Idents.get("__builtin_trap");
    DeclarationName TrapName(&TrapII);
    LookupResult R(SemaRef, TrapName, SourceLocation(), Sema::LookupOrdinaryName);
    if (SemaRef.LookupBuiltin(R))
      TrapFn = dyn_cast_or_null<FunctionDecl>(R.getFoundDecl());

    if (TrapFn) {
      ExprResult TrapCallee = SemaRef.BuildDeclRefExpr(
          TrapFn, TrapFn->getType(), VK_LValue, SourceLocation());
      if (!TrapCallee.isInvalid()) {
        SmallVector<Expr*,0> TrapArgs;
        ExprResult TrapCall = SemaRef.BuildCallExpr(
            nullptr, TrapCallee.get(), SourceLocation(), TrapArgs, SourceLocation());
        if (!TrapCall.isInvalid()) {
          SmallVector<Stmt*,1> DStmts = {TrapCall.get()};
          CompoundStmt *DBody = CompoundStmt::Create(
              Ctx, DStmts, FPOptionsOverride(),
              SourceLocation(), SourceLocation());
          Default = new (Ctx) DefaultStmt(SourceLocation(), SourceLocation(), DBody);
          Default->setNextSwitchCase(CaseList);
          CaseList = Default;
          TopLevelCaseStmts.push_back(Default);
        }
      }
    }
  }

  // ---- switch body ----
  CompoundStmt *SwitchBody = CompoundStmt::Create(
      Ctx, TopLevelCaseStmts, FPOptionsOverride(),
      SourceLocation(), SourceLocation());
  Switch->setBody(SwitchBody);
  Switch->setSwitchCaseList(CaseList);

  // State machine function body
  SmallVector<Stmt *, 4> Statements;
  Statements.push_back(SelfTypedDeclStmt);
  if (ChildCountDeclStmt)
    Statements.push_back(ChildCountDeclStmt);
  Statements.push_back(Switch);
  Statements.push_back(ReturnStmt::Create(Ctx, SourceLocation(), nullptr, nullptr));

  CompoundStmt *NewBody = CompoundStmt::Create(
      Ctx, Statements, FPOptionsOverride(), Body->getLBracLoc(), Body->getRBracLoc());
  StateMachineFD->setBody(NewBody);

  // top-level decl
  Decl *D = StateMachineFD;
  DeclGroupRef DG = DeclGroupRef::Create(Ctx, &D, 1);
  SemaRef.getASTConsumer().HandleTopLevelDecl(DG);

  // original FD is stubbed
  SmallVector<Stmt*, 2> StubStmts;
  StubStmts.push_back(ReturnStmt::Create(Ctx, SourceLocation(), nullptr, nullptr));
  CompoundStmt *EmptyBody = CompoundStmt::Create(
      Ctx, StubStmts, FPOptionsOverride(), Body->getLBracLoc(), Body->getRBracLoc());
  FD->setBody(EmptyBody);

  return StmtResult(EmptyBody);
}
