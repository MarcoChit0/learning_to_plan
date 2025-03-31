; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 792962

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child253 child481 child451 - child
    bread326 bread381 bread404 - bread-portion
    content190 content113 content245 - content-portion
    tray352 tray74 tray167 - tray
    table53 table281 table129 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray352 kitchen)
     (at tray74 kitchen)
     (at tray167 kitchen)
     (at_kitchen_bread bread326)
     (at_kitchen_bread bread381)
     (at_kitchen_bread bread404)
     (at_kitchen_content content190)
     (at_kitchen_content content113)
     (at_kitchen_content content245)
     (not_allergic_gluten child253)
     (not_allergic_gluten child451)
     (not_allergic_gluten child481)
     (waiting child253 table281)
     (waiting child481 table53)
     (waiting child451 table129)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child253)
     (served child481)
     (served child451)
    )
  )
)
