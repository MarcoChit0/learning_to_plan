; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.9
; random seed: 922975

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child10 child80 - child
    bread279 bread22 - bread-portion
    content443 content486 - content-portion
    tray445 tray218 tray353 tray239 - tray
    table80 table464 table130 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray445 kitchen)
     (at tray218 kitchen)
     (at tray353 kitchen)
     (at tray239 kitchen)
     (at_kitchen_bread bread279)
     (at_kitchen_bread bread22)
     (at_kitchen_content content443)
     (at_kitchen_content content486)
     (no_gluten_bread bread22)
     (no_gluten_content content443)
     (allergic_gluten child80)
     (not_allergic_gluten child10)
     (waiting child10 table80)
     (waiting child80 table464)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child10)
     (served child80)
    )
  )
)
