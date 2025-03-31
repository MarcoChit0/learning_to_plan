; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 329210

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child73 - child
    bread454 - bread-portion
    content316 - content-portion
    tray491 - tray
    table321 table419 table452 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray491 kitchen)
     (at_kitchen_bread bread454)
     (at_kitchen_content content316)
     (not_allergic_gluten child73)
     (waiting child73 table321)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child73)
    )
  )
)
