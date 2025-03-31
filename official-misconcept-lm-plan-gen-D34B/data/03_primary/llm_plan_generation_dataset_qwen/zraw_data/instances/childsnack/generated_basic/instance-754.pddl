; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 398256

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child196 child182 - child
    bread499 bread316 - bread-portion
    content361 content80 - content-portion
    tray296 tray90 - tray
    table11 table200 table9 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray296 kitchen)
     (at tray90 kitchen)
     (at_kitchen_bread bread499)
     (at_kitchen_bread bread316)
     (at_kitchen_content content361)
     (at_kitchen_content content80)
     (no_gluten_bread bread499)
     (no_gluten_content content80)
     (allergic_gluten child182)
     (not_allergic_gluten child196)
     (waiting child196 table9)
     (waiting child182 table200)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child196)
     (served child182)
    )
  )
)
