; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 114838

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child421 - child
    bread399 - bread-portion
    content143 - content-portion
    tray422 tray172 tray112 - tray
    table464 table454 table427 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray422 kitchen)
     (at tray172 kitchen)
     (at tray112 kitchen)
     (at_kitchen_bread bread399)
     (at_kitchen_content content143)
     (not_allergic_gluten child421)
     (waiting child421 table464)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child421)
    )
  )
)
