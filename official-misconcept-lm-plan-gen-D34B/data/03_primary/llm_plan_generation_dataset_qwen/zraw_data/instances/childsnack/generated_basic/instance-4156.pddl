; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 547759

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child274 child182 - child
    bread322 bread393 - bread-portion
    content429 content334 - content-portion
    tray100 tray274 tray78 - tray
    table297 table253 table92 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray100 kitchen)
     (at tray274 kitchen)
     (at tray78 kitchen)
     (at_kitchen_bread bread322)
     (at_kitchen_bread bread393)
     (at_kitchen_content content429)
     (at_kitchen_content content334)
     (not_allergic_gluten child274)
     (not_allergic_gluten child182)
     (waiting child274 table253)
     (waiting child182 table297)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child274)
     (served child182)
    )
  )
)
