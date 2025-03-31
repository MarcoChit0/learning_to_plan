; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 325389

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child420 child98 - child
    bread18 bread237 - bread-portion
    content299 content140 - content-portion
    tray403 - tray
    table330 table230 table442 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray403 kitchen)
     (at_kitchen_bread bread18)
     (at_kitchen_bread bread237)
     (at_kitchen_content content299)
     (at_kitchen_content content140)
     (no_gluten_bread bread237)
     (no_gluten_content content299)
     (allergic_gluten child420)
     (not_allergic_gluten child98)
     (waiting child420 table230)
     (waiting child98 table330)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child420)
     (served child98)
    )
  )
)
