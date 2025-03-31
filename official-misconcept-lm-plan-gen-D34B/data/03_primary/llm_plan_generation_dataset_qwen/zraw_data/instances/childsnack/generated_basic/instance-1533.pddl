; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 922660

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child290 child171 - child
    bread480 bread479 - bread-portion
    content469 content207 - content-portion
    tray288 tray6 tray239 tray275 - tray
    table132 table193 table420 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray288 kitchen)
     (at tray6 kitchen)
     (at tray239 kitchen)
     (at tray275 kitchen)
     (at_kitchen_bread bread480)
     (at_kitchen_bread bread479)
     (at_kitchen_content content469)
     (at_kitchen_content content207)
     (not_allergic_gluten child290)
     (not_allergic_gluten child171)
     (waiting child290 table420)
     (waiting child171 table193)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child290)
     (served child171)
    )
  )
)
