; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 244795

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child218 - child
    bread427 - bread-portion
    content470 - content-portion
    tray230 tray248 tray14 tray480 - tray
    table174 table108 table424 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray230 kitchen)
     (at tray248 kitchen)
     (at tray14 kitchen)
     (at tray480 kitchen)
     (at_kitchen_bread bread427)
     (at_kitchen_content content470)
     (not_allergic_gluten child218)
     (waiting child218 table424)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child218)
    )
  )
)
