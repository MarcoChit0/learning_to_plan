; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 347818

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child114 - child
    bread192 - bread-portion
    content82 - content-portion
    tray24 tray415 tray28 tray432 - tray
    table242 table388 table274 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray24 kitchen)
     (at tray415 kitchen)
     (at tray28 kitchen)
     (at tray432 kitchen)
     (at_kitchen_bread bread192)
     (at_kitchen_content content82)
     (not_allergic_gluten child114)
     (waiting child114 table242)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child114)
    )
  )
)
