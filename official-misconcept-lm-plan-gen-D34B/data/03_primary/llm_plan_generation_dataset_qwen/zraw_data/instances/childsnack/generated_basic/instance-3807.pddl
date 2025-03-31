; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 780384

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child112 - child
    bread494 - bread-portion
    content157 - content-portion
    tray312 - tray
    table75 table13 table66 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray312 kitchen)
     (at_kitchen_bread bread494)
     (at_kitchen_content content157)
     (not_allergic_gluten child112)
     (waiting child112 table13)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child112)
    )
  )
)
