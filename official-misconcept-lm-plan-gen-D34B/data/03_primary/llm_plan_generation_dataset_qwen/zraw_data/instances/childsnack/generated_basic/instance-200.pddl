; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 669057

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child283 child412 - child
    bread409 bread308 - bread-portion
    content396 content162 - content-portion
    tray446 tray158 tray143 - tray
    table237 table275 table311 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray446 kitchen)
     (at tray158 kitchen)
     (at tray143 kitchen)
     (at_kitchen_bread bread409)
     (at_kitchen_bread bread308)
     (at_kitchen_content content396)
     (at_kitchen_content content162)
     (not_allergic_gluten child412)
     (not_allergic_gluten child283)
     (waiting child283 table237)
     (waiting child412 table275)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child283)
     (served child412)
    )
  )
)
