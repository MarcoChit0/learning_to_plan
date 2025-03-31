; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 125285

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child164 child16 - child
    bread284 bread176 - bread-portion
    content160 content385 - content-portion
    tray4 tray303 - tray
    table422 table303 table434 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray4 kitchen)
     (at tray303 kitchen)
     (at_kitchen_bread bread284)
     (at_kitchen_bread bread176)
     (at_kitchen_content content160)
     (at_kitchen_content content385)
     (no_gluten_bread bread284)
     (no_gluten_content content160)
     (allergic_gluten child16)
     (not_allergic_gluten child164)
     (waiting child164 table303)
     (waiting child16 table434)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child164)
     (served child16)
    )
  )
)
