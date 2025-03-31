; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 587302

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child50 - child
    bread385 - bread-portion
    content207 - content-portion
    tray390 tray208 tray86 tray13 - tray
    table360 table106 table213 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray390 kitchen)
     (at tray208 kitchen)
     (at tray86 kitchen)
     (at tray13 kitchen)
     (at_kitchen_bread bread385)
     (at_kitchen_content content207)
     (not_allergic_gluten child50)
     (waiting child50 table213)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child50)
    )
  )
)
