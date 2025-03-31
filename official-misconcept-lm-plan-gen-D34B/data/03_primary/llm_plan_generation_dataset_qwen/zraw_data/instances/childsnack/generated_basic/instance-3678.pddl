; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 37384

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child132 child426 - child
    bread247 bread154 - bread-portion
    content450 content71 - content-portion
    tray184 tray412 - tray
    table9 table172 table268 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray184 kitchen)
     (at tray412 kitchen)
     (at_kitchen_bread bread247)
     (at_kitchen_bread bread154)
     (at_kitchen_content content450)
     (at_kitchen_content content71)
     (no_gluten_bread bread154)
     (no_gluten_content content450)
     (allergic_gluten child132)
     (not_allergic_gluten child426)
     (waiting child132 table172)
     (waiting child426 table9)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child132)
     (served child426)
    )
  )
)
