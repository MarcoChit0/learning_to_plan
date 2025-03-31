; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 438169

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child415 child331 - child
    bread308 bread378 - bread-portion
    content43 content149 - content-portion
    tray359 - tray
    table442 table265 table358 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray359 kitchen)
     (at_kitchen_bread bread308)
     (at_kitchen_bread bread378)
     (at_kitchen_content content43)
     (at_kitchen_content content149)
     (no_gluten_bread bread378)
     (no_gluten_content content149)
     (allergic_gluten child415)
     (not_allergic_gluten child331)
     (waiting child415 table442)
     (waiting child331 table265)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child415)
     (served child331)
    )
  )
)
