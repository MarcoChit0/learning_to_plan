; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 173085

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child284 child82 - child
    bread47 bread182 - bread-portion
    content171 content24 - content-portion
    tray363 tray196 tray207 - tray
    table131 table11 table422 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray363 kitchen)
     (at tray196 kitchen)
     (at tray207 kitchen)
     (at_kitchen_bread bread47)
     (at_kitchen_bread bread182)
     (at_kitchen_content content171)
     (at_kitchen_content content24)
     (no_gluten_bread bread47)
     (no_gluten_content content24)
     (allergic_gluten child82)
     (not_allergic_gluten child284)
     (waiting child284 table422)
     (waiting child82 table11)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child284)
     (served child82)
    )
  )
)
