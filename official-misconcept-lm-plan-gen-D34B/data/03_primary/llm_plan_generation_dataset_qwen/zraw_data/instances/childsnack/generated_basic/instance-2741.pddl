; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 611513

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child232 child230 - child
    bread292 bread88 - bread-portion
    content30 content499 - content-portion
    tray334 - tray
    table459 table363 table183 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray334 kitchen)
     (at_kitchen_bread bread292)
     (at_kitchen_bread bread88)
     (at_kitchen_content content30)
     (at_kitchen_content content499)
     (no_gluten_bread bread88)
     (no_gluten_content content30)
     (allergic_gluten child232)
     (not_allergic_gluten child230)
     (waiting child232 table183)
     (waiting child230 table363)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child232)
     (served child230)
    )
  )
)
