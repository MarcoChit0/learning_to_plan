; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 380851

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child214 child196 - child
    bread377 bread46 - bread-portion
    content227 content166 - content-portion
    tray439 tray394 - tray
    table486 table66 table340 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray439 kitchen)
     (at tray394 kitchen)
     (at_kitchen_bread bread377)
     (at_kitchen_bread bread46)
     (at_kitchen_content content227)
     (at_kitchen_content content166)
     (no_gluten_bread bread46)
     (no_gluten_content content227)
     (allergic_gluten child214)
     (not_allergic_gluten child196)
     (waiting child214 table486)
     (waiting child196 table340)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child214)
     (served child196)
    )
  )
)
