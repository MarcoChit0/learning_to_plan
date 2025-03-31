; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 229802

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child52 child140 - child
    bread451 bread476 - bread-portion
    content478 content357 - content-portion
    tray50 - tray
    table475 table143 table279 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray50 kitchen)
     (at_kitchen_bread bread451)
     (at_kitchen_bread bread476)
     (at_kitchen_content content478)
     (at_kitchen_content content357)
     (not_allergic_gluten child140)
     (not_allergic_gluten child52)
     (waiting child52 table143)
     (waiting child140 table279)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child52)
     (served child140)
    )
  )
)
