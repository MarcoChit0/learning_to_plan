; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 341300

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child396 child340 - child
    bread316 bread262 - bread-portion
    content103 content307 - content-portion
    tray320 - tray
    table148 table271 table91 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray320 kitchen)
     (at_kitchen_bread bread316)
     (at_kitchen_bread bread262)
     (at_kitchen_content content103)
     (at_kitchen_content content307)
     (no_gluten_bread bread262)
     (no_gluten_content content307)
     (allergic_gluten child396)
     (not_allergic_gluten child340)
     (waiting child396 table91)
     (waiting child340 table271)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child396)
     (served child340)
    )
  )
)
