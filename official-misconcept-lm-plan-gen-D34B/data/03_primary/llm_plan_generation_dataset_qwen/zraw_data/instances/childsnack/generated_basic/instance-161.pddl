; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 563092

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child451 child361 - child
    bread11 bread247 - bread-portion
    content101 content309 - content-portion
    tray209 tray311 tray327 tray123 - tray
    table367 table5 table475 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray209 kitchen)
     (at tray311 kitchen)
     (at tray327 kitchen)
     (at tray123 kitchen)
     (at_kitchen_bread bread11)
     (at_kitchen_bread bread247)
     (at_kitchen_content content101)
     (at_kitchen_content content309)
     (not_allergic_gluten child451)
     (not_allergic_gluten child361)
     (waiting child451 table5)
     (waiting child361 table5)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child451)
     (served child361)
    )
  )
)
