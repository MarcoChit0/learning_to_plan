; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 508097

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child378 child98 - child
    bread428 bread466 - bread-portion
    content496 content200 - content-portion
    tray403 tray417 - tray
    table461 table393 table0 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray403 kitchen)
     (at tray417 kitchen)
     (at_kitchen_bread bread428)
     (at_kitchen_bread bread466)
     (at_kitchen_content content496)
     (at_kitchen_content content200)
     (no_gluten_bread bread428)
     (no_gluten_content content200)
     (allergic_gluten child378)
     (not_allergic_gluten child98)
     (waiting child378 table0)
     (waiting child98 table393)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child378)
     (served child98)
    )
  )
)
