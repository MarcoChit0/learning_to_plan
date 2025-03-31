; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 354826

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child492 child390 - child
    bread169 bread328 - bread-portion
    content476 content215 - content-portion
    tray494 tray117 - tray
    table407 table199 table6 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray494 kitchen)
     (at tray117 kitchen)
     (at_kitchen_bread bread169)
     (at_kitchen_bread bread328)
     (at_kitchen_content content476)
     (at_kitchen_content content215)
     (no_gluten_bread bread169)
     (no_gluten_content content215)
     (allergic_gluten child492)
     (not_allergic_gluten child390)
     (waiting child492 table6)
     (waiting child390 table407)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child492)
     (served child390)
    )
  )
)
