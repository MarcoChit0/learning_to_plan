; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 66029

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child79 - child
    bread263 - bread-portion
    content484 - content-portion
    tray106 tray177 tray393 - tray
    table222 table347 table425 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray106 kitchen)
     (at tray177 kitchen)
     (at tray393 kitchen)
     (at_kitchen_bread bread263)
     (at_kitchen_content content484)
     (not_allergic_gluten child79)
     (waiting child79 table347)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child79)
    )
  )
)
