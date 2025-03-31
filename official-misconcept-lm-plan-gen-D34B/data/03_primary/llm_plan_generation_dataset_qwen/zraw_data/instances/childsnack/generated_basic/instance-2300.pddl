; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 385893

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child348 child347 - child
    bread197 bread455 - bread-portion
    content70 content49 - content-portion
    tray379 - tray
    table365 table287 table142 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray379 kitchen)
     (at_kitchen_bread bread197)
     (at_kitchen_bread bread455)
     (at_kitchen_content content70)
     (at_kitchen_content content49)
     (no_gluten_bread bread455)
     (no_gluten_content content49)
     (allergic_gluten child347)
     (not_allergic_gluten child348)
     (waiting child348 table142)
     (waiting child347 table287)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child348)
     (served child347)
    )
  )
)
