; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 635436

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child261 - child
    bread174 - bread-portion
    content389 - content-portion
    tray349 tray171 tray26 - tray
    table431 table261 table471 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray349 kitchen)
     (at tray171 kitchen)
     (at tray26 kitchen)
     (at_kitchen_bread bread174)
     (at_kitchen_content content389)
     (not_allergic_gluten child261)
     (waiting child261 table431)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child261)
    )
  )
)
