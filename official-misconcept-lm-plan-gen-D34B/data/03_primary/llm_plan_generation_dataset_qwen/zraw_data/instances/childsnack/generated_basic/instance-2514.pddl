; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 103179

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child50 - child
    bread91 - bread-portion
    content354 - content-portion
    tray114 tray437 tray99 - tray
    table165 table206 table14 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray114 kitchen)
     (at tray437 kitchen)
     (at tray99 kitchen)
     (at_kitchen_bread bread91)
     (at_kitchen_content content354)
     (not_allergic_gluten child50)
     (waiting child50 table14)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child50)
    )
  )
)
