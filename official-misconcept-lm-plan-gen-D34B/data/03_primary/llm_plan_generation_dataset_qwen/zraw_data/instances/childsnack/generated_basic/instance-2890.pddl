; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 63885

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child177 child95 - child
    bread89 bread31 - bread-portion
    content355 content277 - content-portion
    tray131 tray177 tray366 - tray
    table306 table223 table62 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray131 kitchen)
     (at tray177 kitchen)
     (at tray366 kitchen)
     (at_kitchen_bread bread89)
     (at_kitchen_bread bread31)
     (at_kitchen_content content355)
     (at_kitchen_content content277)
     (no_gluten_bread bread31)
     (no_gluten_content content277)
     (allergic_gluten child95)
     (not_allergic_gluten child177)
     (waiting child177 table62)
     (waiting child95 table223)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child177)
     (served child95)
    )
  )
)
