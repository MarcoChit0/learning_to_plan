; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 379377

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child351 - child
    bread110 - bread-portion
    content170 - content-portion
    tray389 tray221 tray70 tray350 - tray
    table265 table56 table252 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray389 kitchen)
     (at tray221 kitchen)
     (at tray70 kitchen)
     (at tray350 kitchen)
     (at_kitchen_bread bread110)
     (at_kitchen_content content170)
     (not_allergic_gluten child351)
     (waiting child351 table252)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child351)
    )
  )
)
