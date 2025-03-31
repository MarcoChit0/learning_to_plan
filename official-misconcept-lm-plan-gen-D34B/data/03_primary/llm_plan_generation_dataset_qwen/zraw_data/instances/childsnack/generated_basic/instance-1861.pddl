; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 199299

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child167 child453 - child
    bread262 bread269 - bread-portion
    content345 content258 - content-portion
    tray437 tray329 - tray
    table130 table78 table112 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray437 kitchen)
     (at tray329 kitchen)
     (at_kitchen_bread bread262)
     (at_kitchen_bread bread269)
     (at_kitchen_content content345)
     (at_kitchen_content content258)
     (no_gluten_bread bread269)
     (no_gluten_content content258)
     (allergic_gluten child453)
     (not_allergic_gluten child167)
     (waiting child167 table130)
     (waiting child453 table112)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child167)
     (served child453)
    )
  )
)
