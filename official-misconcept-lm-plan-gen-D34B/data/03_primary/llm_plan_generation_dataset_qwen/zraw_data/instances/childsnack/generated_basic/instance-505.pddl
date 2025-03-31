; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 972267

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child248 child245 - child
    bread176 bread115 - bread-portion
    content53 content171 - content-portion
    tray236 tray96 - tray
    table181 table370 table193 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray236 kitchen)
     (at tray96 kitchen)
     (at_kitchen_bread bread176)
     (at_kitchen_bread bread115)
     (at_kitchen_content content53)
     (at_kitchen_content content171)
     (not_allergic_gluten child248)
     (not_allergic_gluten child245)
     (waiting child248 table181)
     (waiting child245 table193)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child248)
     (served child245)
    )
  )
)
