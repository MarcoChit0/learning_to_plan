; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 556003

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child356 - child
    bread291 - bread-portion
    content234 - content-portion
    tray497 tray172 tray148 tray126 - tray
    table164 table80 table179 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray497 kitchen)
     (at tray172 kitchen)
     (at tray148 kitchen)
     (at tray126 kitchen)
     (at_kitchen_bread bread291)
     (at_kitchen_content content234)
     (not_allergic_gluten child356)
     (waiting child356 table164)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child356)
    )
  )
)
