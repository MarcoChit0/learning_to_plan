; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 822664

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child316 - child
    bread484 - bread-portion
    content245 - content-portion
    tray484 tray13 tray499 tray265 - tray
    table439 table356 table398 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray484 kitchen)
     (at tray13 kitchen)
     (at tray499 kitchen)
     (at tray265 kitchen)
     (at_kitchen_bread bread484)
     (at_kitchen_content content245)
     (not_allergic_gluten child316)
     (waiting child316 table356)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child316)
    )
  )
)
