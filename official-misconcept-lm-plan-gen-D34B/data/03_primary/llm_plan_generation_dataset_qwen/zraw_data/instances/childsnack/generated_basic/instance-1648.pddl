; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 913617

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child386 - child
    bread458 - bread-portion
    content235 - content-portion
    tray176 tray321 tray102 tray448 - tray
    table15 table423 table109 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray176 kitchen)
     (at tray321 kitchen)
     (at tray102 kitchen)
     (at tray448 kitchen)
     (at_kitchen_bread bread458)
     (at_kitchen_content content235)
     (not_allergic_gluten child386)
     (waiting child386 table423)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child386)
    )
  )
)
