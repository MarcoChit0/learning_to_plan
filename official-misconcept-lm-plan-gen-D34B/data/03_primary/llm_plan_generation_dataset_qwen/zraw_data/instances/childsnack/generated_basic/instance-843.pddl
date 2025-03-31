; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 977787

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child242 child94 - child
    bread39 bread304 - bread-portion
    content451 content450 - content-portion
    tray70 tray191 - tray
    table244 table66 table218 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray70 kitchen)
     (at tray191 kitchen)
     (at_kitchen_bread bread39)
     (at_kitchen_bread bread304)
     (at_kitchen_content content451)
     (at_kitchen_content content450)
     (no_gluten_bread bread39)
     (no_gluten_content content450)
     (allergic_gluten child94)
     (not_allergic_gluten child242)
     (waiting child242 table66)
     (waiting child94 table244)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child242)
     (served child94)
    )
  )
)
