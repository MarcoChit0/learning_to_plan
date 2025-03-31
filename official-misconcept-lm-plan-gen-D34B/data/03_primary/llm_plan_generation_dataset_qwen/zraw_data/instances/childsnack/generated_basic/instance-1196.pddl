; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 565743

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child475 child167 child149 - child
    bread487 bread149 bread59 - bread-portion
    content304 content397 content411 - content-portion
    tray119 tray97 tray58 - tray
    table269 table190 table214 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray119 kitchen)
     (at tray97 kitchen)
     (at tray58 kitchen)
     (at_kitchen_bread bread487)
     (at_kitchen_bread bread149)
     (at_kitchen_bread bread59)
     (at_kitchen_content content304)
     (at_kitchen_content content397)
     (at_kitchen_content content411)
     (not_allergic_gluten child475)
     (not_allergic_gluten child167)
     (not_allergic_gluten child149)
     (waiting child475 table269)
     (waiting child167 table214)
     (waiting child149 table214)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child475)
     (served child167)
     (served child149)
    )
  )
)
