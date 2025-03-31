; child-snack task with 3 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 98488

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child345 child106 child172 - child
    bread431 bread333 bread363 - bread-portion
    content213 content441 content182 - content-portion
    tray245 tray398 tray5 tray239 - tray
    table203 table232 table273 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray245 kitchen)
     (at tray398 kitchen)
     (at tray5 kitchen)
     (at tray239 kitchen)
     (at_kitchen_bread bread431)
     (at_kitchen_bread bread333)
     (at_kitchen_bread bread363)
     (at_kitchen_content content213)
     (at_kitchen_content content441)
     (at_kitchen_content content182)
     (no_gluten_bread bread333)
     (no_gluten_content content441)
     (allergic_gluten child172)
     (not_allergic_gluten child345)
     (not_allergic_gluten child106)
     (waiting child345 table273)
     (waiting child106 table273)
     (waiting child172 table273)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child345)
     (served child106)
     (served child172)
    )
  )
)
