; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 290437

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child162 - child
    bread54 - bread-portion
    content494 - content-portion
    tray123 - tray
    table41 table47 table124 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray123 kitchen)
     (at_kitchen_bread bread54)
     (at_kitchen_content content494)
     (not_allergic_gluten child162)
     (waiting child162 table47)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child162)
    )
  )
)
