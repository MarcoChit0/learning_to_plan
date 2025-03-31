; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 152992

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child128 - child
    bread85 - bread-portion
    content456 - content-portion
    tray422 - tray
    table387 table46 table136 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray422 kitchen)
     (at_kitchen_bread bread85)
     (at_kitchen_content content456)
     (not_allergic_gluten child128)
     (waiting child128 table387)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child128)
    )
  )
)
