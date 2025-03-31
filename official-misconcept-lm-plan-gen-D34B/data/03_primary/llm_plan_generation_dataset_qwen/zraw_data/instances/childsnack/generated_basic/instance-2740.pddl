; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 354238

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child453 child268 - child
    bread486 bread374 - bread-portion
    content31 content35 - content-portion
    tray260 - tray
    table445 table216 table25 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray260 kitchen)
     (at_kitchen_bread bread486)
     (at_kitchen_bread bread374)
     (at_kitchen_content content31)
     (at_kitchen_content content35)
     (no_gluten_bread bread374)
     (no_gluten_content content31)
     (allergic_gluten child453)
     (not_allergic_gluten child268)
     (waiting child453 table216)
     (waiting child268 table445)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child453)
     (served child268)
    )
  )
)
