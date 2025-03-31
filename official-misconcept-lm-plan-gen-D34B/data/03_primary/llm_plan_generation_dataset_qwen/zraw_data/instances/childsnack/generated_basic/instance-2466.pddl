; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 284334

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child385 child304 - child
    bread56 bread303 - bread-portion
    content302 content245 - content-portion
    tray369 tray337 - tray
    table259 table261 table243 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray369 kitchen)
     (at tray337 kitchen)
     (at_kitchen_bread bread56)
     (at_kitchen_bread bread303)
     (at_kitchen_content content302)
     (at_kitchen_content content245)
     (not_allergic_gluten child385)
     (not_allergic_gluten child304)
     (waiting child385 table259)
     (waiting child304 table243)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child385)
     (served child304)
    )
  )
)
