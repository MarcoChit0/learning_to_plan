; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 37817

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child238 child401 - child
    bread385 bread292 - bread-portion
    content153 content213 - content-portion
    tray151 - tray
    table173 table427 table224 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray151 kitchen)
     (at_kitchen_bread bread385)
     (at_kitchen_bread bread292)
     (at_kitchen_content content153)
     (at_kitchen_content content213)
     (no_gluten_bread bread292)
     (no_gluten_content content213)
     (allergic_gluten child238)
     (not_allergic_gluten child401)
     (waiting child238 table224)
     (waiting child401 table173)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child238)
     (served child401)
    )
  )
)
