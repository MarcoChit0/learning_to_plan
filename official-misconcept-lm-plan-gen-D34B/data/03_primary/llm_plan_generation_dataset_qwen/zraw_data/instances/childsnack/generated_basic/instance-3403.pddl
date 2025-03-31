; child-snack task with 3 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 699599

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child232 child340 child247 - child
    bread394 bread174 bread411 - bread-portion
    content452 content489 content342 - content-portion
    tray370 tray47 tray265 - tray
    table163 table486 table204 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray370 kitchen)
     (at tray47 kitchen)
     (at tray265 kitchen)
     (at_kitchen_bread bread394)
     (at_kitchen_bread bread174)
     (at_kitchen_bread bread411)
     (at_kitchen_content content452)
     (at_kitchen_content content489)
     (at_kitchen_content content342)
     (no_gluten_bread bread411)
     (no_gluten_content content452)
     (allergic_gluten child247)
     (not_allergic_gluten child232)
     (not_allergic_gluten child340)
     (waiting child232 table163)
     (waiting child340 table163)
     (waiting child247 table163)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child232)
     (served child340)
     (served child247)
    )
  )
)
