; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 140019

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child262 child298 - child
    bread164 bread385 - bread-portion
    content299 content402 - content-portion
    tray362 tray407 - tray
    table14 table29 table281 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray362 kitchen)
     (at tray407 kitchen)
     (at_kitchen_bread bread164)
     (at_kitchen_bread bread385)
     (at_kitchen_content content299)
     (at_kitchen_content content402)
     (no_gluten_bread bread385)
     (no_gluten_content content299)
     (allergic_gluten child298)
     (not_allergic_gluten child262)
     (waiting child262 table14)
     (waiting child298 table29)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child262)
     (served child298)
    )
  )
)
