; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 660816

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child65 child447 - child
    bread403 bread182 - bread-portion
    content459 content309 - content-portion
    tray356 tray122 tray0 tray309 - tray
    table284 table325 table195 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray356 kitchen)
     (at tray122 kitchen)
     (at tray0 kitchen)
     (at tray309 kitchen)
     (at_kitchen_bread bread403)
     (at_kitchen_bread bread182)
     (at_kitchen_content content459)
     (at_kitchen_content content309)
     (no_gluten_bread bread182)
     (no_gluten_content content309)
     (allergic_gluten child447)
     (not_allergic_gluten child65)
     (waiting child65 table195)
     (waiting child447 table325)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child65)
     (served child447)
    )
  )
)
