; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 42021

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child421 - child
    bread34 - bread-portion
    content461 - content-portion
    tray306 tray454 tray319 tray396 - tray
    table430 table263 table102 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray306 kitchen)
     (at tray454 kitchen)
     (at tray319 kitchen)
     (at tray396 kitchen)
     (at_kitchen_bread bread34)
     (at_kitchen_content content461)
     (not_allergic_gluten child421)
     (waiting child421 table263)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child421)
    )
  )
)
