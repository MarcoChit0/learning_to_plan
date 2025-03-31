; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 212204

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child132 - child
    bread342 - bread-portion
    content112 - content-portion
    tray132 tray398 tray228 - tray
    table156 table419 table463 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray132 kitchen)
     (at tray398 kitchen)
     (at tray228 kitchen)
     (at_kitchen_bread bread342)
     (at_kitchen_content content112)
     (not_allergic_gluten child132)
     (waiting child132 table463)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child132)
    )
  )
)
