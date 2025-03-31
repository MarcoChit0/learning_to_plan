; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 540897

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child80 - child
    bread6 - bread-portion
    content349 - content-portion
    tray379 - tray
    table354 table442 table124 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray379 kitchen)
     (at_kitchen_bread bread6)
     (at_kitchen_content content349)
     (not_allergic_gluten child80)
     (waiting child80 table354)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child80)
    )
  )
)
