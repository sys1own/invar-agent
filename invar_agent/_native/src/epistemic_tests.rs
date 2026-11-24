use std::collections::{BTreeMap, HashMap};

use crate::{adjoint::*, his::*, holographic_hin::*, uast::*};

fn var(s: &str) -> CselExpr { CselExpr::Symbol { name: s.into() } }
fn num(value: f64) -> CselExpr { CselExpr::Literal { value } }
fn bin(op: &str, left: CselExpr, right: CselExpr) -> CselExpr {
    CselExpr::Binary { op: op.into(), left: Box::new(left), right: Box::new(right) }
}
fn text(e: &CselExpr) -> String { serde_json::to_string(e).unwrap() }
fn sample(k: u64) -> f64 {
    let mut x = k.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    ((x ^ (x >> 31)) >> 11) as f64 / 9007199254740992.0
}

#[test]
fn commitments_are_canonical_and_domain_separated() {
    assert_eq!(epistemic_canonical_bytes(r#"{"x":-0.0,"r":{"numerator":2,"denominator":4}}"#).unwrap(),
        epistemic_canonical_bytes(r#"{"r":{"denominator":2,"numerator":1},"x":0.0}"#).unwrap());
    assert!(epistemic_canonical_bytes(r#"{"numerator":0,"denominator":0}"#).is_err());
    let a = compute_ihcg_digest(r#"{"a":1,"h_ihcg":"ignored"}"#).unwrap();
    assert_eq!(a,compute_ihcg_digest(r#"{"h_ihcg":"different","a":1}"#).unwrap());
    assert_ne!(a,compute_ihcg_stage_chain(&a,r#"{"a":1}"#).unwrap());
    assert!(compute_ihcg_stage_chain("1234","{}").is_err());
    let c = compute_cds_digests("m","claim","domain"); let d = compute_cds_digests("m","claim","extended");
    assert_eq!(c.claim_hash,d.claim_hash); assert_ne!(c.full_hash,d.full_hash);
    assert_ne!(c.model_hash,c.claim_hash);
    assert_eq!(c.model_hash,blake3::hash(b"invar.cds.model.v1m").to_hex().to_string());
    assert_eq!(compute_certificate_receipt(r#"{"a":1}"#).unwrap(),
        compute_certificate_receipt(r#"{"a":1,"blake3_receipt":"old"}"#).unwrap());
}

fn wire(role: WireRole) -> WireType {
    WireType { mell_type: MELLType::Linear, dimensions: SIDimensionVector::dimensionless(),
        bound: IntervalBound::point(0.0), role, purity: PurityTag::Pure, is_falsification_target: false }
}
fn write(slot_idx: usize, purity: PurityTag) -> SlotWriteOp {
    SlotWriteOp { slot_idx, purity, source_slots: Vec::new(), has_conservation_guard: false,
        instrumented: false, recalibrates_obligation: false }
}

#[test]
fn purity_rules_defaults_and_transitive_noninterference() {
    // Default compatibility is tested using the current dimension serialization.
    let mut old = serde_json::to_value(wire(WireRole::DynamicState)).unwrap();
    old.as_object_mut().unwrap().remove("role"); old.as_object_mut().unwrap().remove("purity"); old.as_object_mut().unwrap().remove("is_falsification_target");
    let restored: WireType = serde_json::from_value(old).unwrap();
    assert_eq!(restored.role,WireRole::DynamicState); assert_eq!(restored.purity,PurityTag::Pure);
    for a in [PurityTag::Pure,PurityTag::Projected,PurityTag::Stabilized,PurityTag::Recalibrated] {
        assert_eq!(a.join(a),a); assert_eq!(a.join(PurityTag::Pure),a);
    }
    let wires = BTreeMap::from([(0,wire(WireRole::ControlParam)),(1,wire(WireRole::DynamicState)),(2,wire(WireRole::DerivedObservable))]);
    assert!(verify_observational_purity(&wires,&[]).unwrap().valid);
    assert!(verify_observational_purity(&wires,&[write(2,PurityTag::Pure)]).is_ok());
    assert_eq!(verify_observational_purity(&wires,&[write(0,PurityTag::Pure)]).unwrap_err()[0].rule,"T-CONST-PARAM");
    assert!(verify_observational_purity(&wires,&[write(1,PurityTag::Projected)]).is_err());
    let mut project = write(1,PurityTag::Projected); project.instrumented=true; project.has_conservation_guard=true;
    assert!(verify_observational_purity(&wires,&[project.clone()]).is_ok());
    let mut observe = write(2,PurityTag::Pure); observe.source_slots=vec![1];
    assert!(verify_observational_purity(&wires,&[observe,project]).unwrap_err().iter().any(|v| v.rule == "T-OBSERVABLE-PURE"));
    assert!(verify_observational_purity(&wires,&[write(1,PurityTag::Recalibrated)]).is_err());
    assert!(verify_observational_purity(&wires,&[write(99,PurityTag::Pure)]).is_err());
}

#[test]
fn jacobi_svd_separates_dead_and_collinear_parameters() {
    let s = compute_sensitivity_svd(&[vec![1.0,0.0],vec![0.0,2.0]],1e-10,1e-10).unwrap();
    assert!((s.singular_values[0]-2.0).abs()<1e-12); assert_eq!(s.rank,2);
    assert!((s.fim_condition_number.unwrap()-4.0).abs()<1e-10);
    let collinear = compute_sensitivity_svd(&[vec![1.0,-1.0],vec![2.0,-2.0]],1e-10,1e-10).unwrap();
    assert!(collinear.active_coupling && collinear.dimensionless_group_collinearity);
    assert!(collinear.brun_gamma_k.is_none());
    let dead = compute_sensitivity_svd(&[vec![1.0,0.0]],1e-10,1e-10).unwrap();
    assert_eq!(dead.dead_parameters,vec![1]); assert!(!dead.active_coupling);
    assert!(compute_sensitivity_svd(&[],1e-10,1e-10).is_err());
    assert!(compute_sensitivity_svd(&[vec![f64::NAN]],1e-10,1e-10).is_err());
    for k in 0..16 {
        let a: Vec<Vec<_>> = (0..4).map(|i| (0..3).map(|j| sample(k*12+i*3+j)-0.5).collect()).collect();
        let report=compute_sensitivity_svd(&a,1e-10,1e-10).unwrap();
        let matrix=nalgebra::DMatrix::from_fn(4,3,|i,j| a[i][j]);
        let ground=matrix.clone().svd(false,false);
        let mut singular: Vec<_>=ground.singular_values.iter().copied().collect(); singular.sort_by(|a,b| b.total_cmp(a));
        for (x,y) in report.singular_values.iter().zip(singular) { assert!((x-y).abs()<1e-10); }
        let v=nalgebra::DMatrix::from_fn(3,3,|i,j| report.right_singular_vectors[i][j]);
        let d=nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(3,report.singular_values.iter().map(|s| s*s)));
        assert!((matrix.transpose()*matrix*&v-&v*d).norm()<1e-10);
        assert!((v.transpose()*v-nalgebra::DMatrix::identity(3,3)).norm()<1e-10);
    }
}

#[test]
fn ift_krawczyk_stability_and_budget_telemetry() {
    let ift=compute_ift_equilibrium_sensitivity(&[vec![2.0]],&[vec![-1.0]],&[vec![3.0]],&[vec![0.0]]).unwrap();
    assert_eq!(ift.dx_dtheta,vec![vec![0.5]]); assert_eq!(ift.dy_dtheta,vec![vec![1.5]]);
    assert!(compute_ift_equilibrium_sensitivity(&[vec![0.0]],&[vec![1.0]],&[vec![1.0]],&[vec![0.0]]).is_err());
    let equation=bin("-",bin("*",var("x0"),var("x0")),num(2.0));
    let k=certify_krawczyk_fixed_point(&[text(&equation)],&[2.0_f64.sqrt()],0.01).unwrap();
    assert!(k.certified); assert!(k.interval_jacobian[0][0].lo < 2.0*2.0_f64.sqrt());
    let excluded=certify_krawczyk_fixed_point(&[text(&bin("-",var("x0"),num(2.0)))],&[0.0],0.1).unwrap();
    assert!(excluded.excluded && !excluded.certified);
    assert!(certify_krawczyk_fixed_point(&[],&[],0.1).is_err());
    for n in 1..=5 {
        let stable: Vec<Vec<_>>=(0..n).map(|i| (0..n).map(|j| IntervalF64::point(if i==j {-1.0} else {0.0})).collect()).collect();
        assert!(certify_interval_stability(&stable).unwrap().certified);
        let unstable: Vec<Vec<_>>=(0..n).map(|i| (0..n).map(|j| IntervalF64::point(if i==j {1.0} else {0.0})).collect()).collect();
        assert!(!certify_interval_stability(&unstable).unwrap().certified);
    }
    let r=branch_and_bound_refute(&text(&bin("-",var("x0"),num(3.0))),&[0.0],&[1.0],0.01,2048).unwrap();
    assert!(r.refuted); assert_eq!(r.certified_refuted_volume_fraction,1.0);
    let r=branch_and_bound_refute(&text(&var("x0")),&[-1.0;5],&[1.0;5],0.0,32).unwrap();
    assert!(!r.refuted && r.budget_exhausted); assert_eq!(r.boxes_evaluated,32);
    assert!(r.certified_refuted_volume_fraction>=0.0 && r.certified_refuted_volume_fraction<=1.0);
    assert!(r.max_unresolved_upper_bound.unwrap()>=0.0);
    let empty_budget=branch_and_bound_refute(&text(&var("x0")),&[-1.0],&[1.0],0.0,0).unwrap();
    assert!(empty_budget.budget_exhausted && empty_budget.unresolved_bound_unbounded && !empty_budget.refuted);
    assert!(branch_and_bound_refute("{}",&[],&[],0.0,1).is_err());
    assert_eq!(dimension_scaled_box_budget(3),2048); assert_eq!(dimension_scaled_box_budget(5),8192); assert_eq!(dimension_scaled_box_budget(100),32768);
}

#[test]
fn symbolic_derivatives_against_1024_finite_difference_points() {
    let x=var("x");
    let mut exprs=vec![bin("+",bin("*",x.clone(),x.clone()),x.clone()),bin("/",x.clone(),bin("+",num(1.0),x.clone()))];
    for name in ["exp","log","sqrt","sin","cos"] { exprs.push(CselExpr::Call {func:name.into(),args:vec![x.clone()]}); }
    for e in exprs {
        let d=certificate_derivative(&e,"x").unwrap();
        for k in 0..1024 {
            let point=0.25+sample(k)*2.0; let h=1e-5;
            let plus=e.eval(&HashMap::from([("x".into(),point+h)]));
            let minus=e.eval(&HashMap::from([("x".into(),point-h)]));
            let finite_difference=(plus-minus)/(2.0*h);
            let env=HashMap::from([("x".into(),IntervalF64::point(point))]);
            let enclosure=eval_interval(&d,&env).unwrap();
            let analytic=0.5*enclosure.lo+0.5*enclosure.hi;
            // Trig intervals are deliberately full codomain; use scalar symbolic evaluation there.
            let analytic=if enclosure.hi-enclosure.lo>0.01 {d.eval(&HashMap::from([("x".into(),point)]))} else {analytic};
            assert!((analytic-finite_difference).abs()<2e-8,"{e:?}: {analytic} vs {finite_difference}");
            assert!(enclosure.lo<=analytic && analytic<=enclosure.hi);
        }
    }
}

#[test]
fn bauer_fike_certifies_non_gershgorin_real_and_complex_eigenbases() {
    let mut triangular=vec![vec![IntervalF64::point(0.0);5];5];
    for i in 0..5 {triangular[i][i]=IntervalF64::point(-(i as f64)-1.0);}
    triangular[0][1]=IntervalF64 {lo:1.19999,hi:1.20001};
    let certificate=certify_interval_stability(&triangular).unwrap();
    assert!(certificate.certified); assert_eq!(certificate.method,"INTERVAL_BAUER_FIKE");
    let bound=certificate.bauer_fike.unwrap();
    assert!(bound.eigenvector_condition_upper>=1.0 && bound.radius_norm_upper>0.0 && bound.spectral_abscissa_upper<0.0);
    triangular[0][0]=IntervalF64::point(-1.0); triangular[1][1]=IntervalF64::point(-1.0);
    triangular[0][1]=IntervalF64 {lo:-2.000001,hi:-1.999999};
    triangular[1][0]=IntervalF64 {lo:1.999999,hi:2.000001};
    let complex=certify_interval_stability(&triangular).unwrap();
    assert!(complex.certified); assert_eq!(complex.method,"INTERVAL_BAUER_FIKE");
    let mut defective=vec![vec![IntervalF64::point(0.0);5];5];
    for i in 0..5 {defective[i][i]=IntervalF64::point(-1.0); if i<4 {defective[i][i+1]=IntervalF64::point(2.0);}}
    let inconclusive=certify_interval_stability(&defective).unwrap();
    assert!(!inconclusive.certified && inconclusive.bauer_fike.is_none());
    let mut ill_conditioned=defective.clone();
    for i in 0..5 {ill_conditioned[i][i]=IntervalF64::point(-1.0+(i as f64)*1e-12);}
    assert!(!certify_interval_stability(&ill_conditioned).unwrap().certified);
    for i in 0..5 {defective[i][i]=IntervalF64::point(-3.0);}
    let gershgorin=certify_interval_stability(&defective).unwrap();
    assert!(gershgorin.certified); assert_eq!(gershgorin.method,"INTERVAL_GERSHGORIN");
    triangular[0][0]=IntervalF64 {lo:-1.0,hi:1.0};
    assert!(!certify_interval_stability(&triangular).unwrap().certified);
    assert!(certify_interval_stability(&[]).is_err());
    assert!(certify_interval_stability(&[vec![IntervalF64 {lo:f64::NAN,hi:1.0}]]).is_err());
}

#[test]
fn arena_metadata_reaches_sources_operators_fanout_and_observable_sinks() {
    let quantities:Vec<QuantitySpec>=serde_json::from_value(serde_json::json!([
        {"symbol":"x","si_dimension":[0,0,0,0,0,0,0],"interval":[0,1],"is_state_variable":true},
        {"symbol":"y","si_dimension":[0,0,0,0,0,0,0],"interval":[0,1],"is_state_variable":false},
        {"symbol":"p","si_dimension":[0,0,0,0,0,0,0],"interval":[0,1],"is_state_variable":false}
    ])).unwrap();
    let mut arena=HolographicHinArena::new(quantities.clone());
    assert_eq!(arena.audit_observational_purity().unwrap().checked_writes,0);
    let expression=CselExpr::Call {func:"sin".into(),args:vec![CselExpr::Unary {op:"-".into(),arg:Box::new(bin("*",var("x"),var("x")))}]};
    let metadata=BTreeMap::from([
        ("x".into(),WireMetadata {purity:PurityTag::Projected,instrumented:true,has_conservation_guard:true,..WireMetadata::default()}),
        ("y".into(),WireMetadata {role:WireRole::DerivedObservable,is_falsification_target:true,..WireMetadata::default()}),
        ("p".into(),WireMetadata {role:WireRole::ControlParam,..WireMetadata::default()})
    ]);
    arena.set_wire_metadata(metadata.clone()).unwrap();
    assert!(arena.verify_equation("y",&expression).valid);
    let violations=arena.audit_observational_purity().unwrap_err();
    assert!(violations.iter().any(|v| v.rule=="T-OBSERVABLE-PURE"));
    assert!(!violations.iter().any(|v| v.rule=="T-STATE-INSTRUMENTED"));
    let wires=arena.wire_types();
    assert_eq!(wires.values().filter(|w| matches!(w.mell_type,MELLType::Exponential(_)) && w.purity==PurityTag::Projected).count(),1);
    assert_eq!(wires.values().filter(|w| w.role==WireRole::DerivedObservable && w.purity==PurityTag::Projected).count(),1);
    assert_eq!(wires.values().filter(|w| w.role==WireRole::DerivedObservable && w.purity==PurityTag::Pure).count(),1);
    assert!(arena.set_wire_metadata(metadata).is_err());
    let mut pure=HolographicHinArena::new(quantities.clone());
    assert!(pure.verify_equation("y",&expression).valid);
    assert!(pure.audit_observational_purity().unwrap().valid);
    let mut cross_quantities=quantities.clone(); let mut z=quantities[1].clone(); z.symbol="z".into(); cross_quantities.push(z);
    let mut cross=HolographicHinArena::new(cross_quantities);
    cross.set_wire_metadata(BTreeMap::from([
        ("x".into(),WireMetadata {purity:PurityTag::Projected,instrumented:true,has_conservation_guard:true,..WireMetadata::default()}),
        ("z".into(),WireMetadata {role:WireRole::DerivedObservable,..WireMetadata::default()})
    ])).unwrap();
    cross.verify_equation("z",&var("y"));
    cross.verify_equation("y",&var("x"));
    assert!(cross.audit_observational_purity().unwrap_err().iter().any(|v| v.rule=="T-OBSERVABLE-PURE"));
    let mut parameter=HolographicHinArena::new(quantities);
    parameter.set_wire_metadata(BTreeMap::from([("p".into(),WireMetadata {role:WireRole::ControlParam,..WireMetadata::default()})])).unwrap();
    parameter.verify_equation("p",&var("x"));
    assert!(parameter.audit_observational_purity().unwrap_err().iter().any(|v| v.rule=="T-CONST-PARAM"));
}
