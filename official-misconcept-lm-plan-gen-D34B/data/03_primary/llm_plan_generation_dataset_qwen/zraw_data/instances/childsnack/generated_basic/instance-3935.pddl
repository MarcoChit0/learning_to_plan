; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 31182

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child479 child370 - child
    bread406 bread196 - bread-portion
    content450 content145 - content-portion
    tray156 - tray
    table234 table14 table426 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray156 kitchen)
     (at_kitchen_bread bread406)
     (at_kitchen_bread bread196)
     (at_kitchen_content content450)
     (at_kitchen_content content145)
     (no_gluten_bread bread196)
     (no_gluten_content content145)
     (allergic_gluten child479)
     (not_allergic_gluten child370)
     (waiting child479 table234)
     (waiting child370 table14)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child479)
     (served child370)
    )
  )
)
