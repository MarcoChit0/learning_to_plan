; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 911563

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child13 child175 - child
    bread490 bread11 - bread-portion
    content430 content206 - content-portion
    tray122 - tray
    table427 table304 table402 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray122 kitchen)
     (at_kitchen_bread bread490)
     (at_kitchen_bread bread11)
     (at_kitchen_content content430)
     (at_kitchen_content content206)
     (no_gluten_bread bread490)
     (no_gluten_content content430)
     (allergic_gluten child13)
     (not_allergic_gluten child175)
     (waiting child13 table304)
     (waiting child175 table402)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child13)
     (served child175)
    )
  )
)
