; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 127310

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child64 child29 - child
    bread271 bread12 - bread-portion
    content195 content404 - content-portion
    tray128 - tray
    table149 table106 table127 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray128 kitchen)
     (at_kitchen_bread bread271)
     (at_kitchen_bread bread12)
     (at_kitchen_content content195)
     (at_kitchen_content content404)
     (no_gluten_bread bread271)
     (no_gluten_content content404)
     (allergic_gluten child64)
     (not_allergic_gluten child29)
     (waiting child64 table106)
     (waiting child29 table106)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child64)
     (served child29)
    )
  )
)
