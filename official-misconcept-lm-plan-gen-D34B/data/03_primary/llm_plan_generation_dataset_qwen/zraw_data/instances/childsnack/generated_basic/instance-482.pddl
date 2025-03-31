; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 72319

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child451 child422 - child
    bread387 bread459 - bread-portion
    content267 content59 - content-portion
    tray401 tray271 - tray
    table498 table42 table259 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray401 kitchen)
     (at tray271 kitchen)
     (at_kitchen_bread bread387)
     (at_kitchen_bread bread459)
     (at_kitchen_content content267)
     (at_kitchen_content content59)
     (not_allergic_gluten child422)
     (not_allergic_gluten child451)
     (waiting child451 table259)
     (waiting child422 table42)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child451)
     (served child422)
    )
  )
)
