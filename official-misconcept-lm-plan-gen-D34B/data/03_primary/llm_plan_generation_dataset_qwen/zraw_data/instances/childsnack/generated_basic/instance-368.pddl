; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 78872

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child394 - child
    bread418 - bread-portion
    content355 - content-portion
    tray336 tray15 tray411 tray135 - tray
    table391 table172 table498 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray336 kitchen)
     (at tray15 kitchen)
     (at tray411 kitchen)
     (at tray135 kitchen)
     (at_kitchen_bread bread418)
     (at_kitchen_content content355)
     (not_allergic_gluten child394)
     (waiting child394 table498)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child394)
    )
  )
)
