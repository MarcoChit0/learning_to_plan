; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 568860

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child420 child188 child18 - child
    bread104 bread110 bread445 - bread-portion
    content276 content50 content8 - content-portion
    tray483 - tray
    table278 table148 table234 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray483 kitchen)
     (at_kitchen_bread bread104)
     (at_kitchen_bread bread110)
     (at_kitchen_bread bread445)
     (at_kitchen_content content276)
     (at_kitchen_content content50)
     (at_kitchen_content content8)
     (not_allergic_gluten child18)
     (not_allergic_gluten child188)
     (not_allergic_gluten child420)
     (waiting child420 table278)
     (waiting child188 table148)
     (waiting child18 table234)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child420)
     (served child188)
     (served child18)
    )
  )
)
