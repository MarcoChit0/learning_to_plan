; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 191626

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child217 child41 - child
    bread467 bread111 - bread-portion
    content173 content254 - content-portion
    tray302 tray58 - tray
    table268 table42 table433 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray302 kitchen)
     (at tray58 kitchen)
     (at_kitchen_bread bread467)
     (at_kitchen_bread bread111)
     (at_kitchen_content content173)
     (at_kitchen_content content254)
     (not_allergic_gluten child41)
     (not_allergic_gluten child217)
     (waiting child217 table268)
     (waiting child41 table42)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child217)
     (served child41)
    )
  )
)
