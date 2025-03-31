; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 689523

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child279 child104 - child
    bread474 bread213 - bread-portion
    content37 content107 - content-portion
    tray413 tray138 - tray
    table491 table279 table20 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray413 kitchen)
     (at tray138 kitchen)
     (at_kitchen_bread bread474)
     (at_kitchen_bread bread213)
     (at_kitchen_content content37)
     (at_kitchen_content content107)
     (not_allergic_gluten child104)
     (not_allergic_gluten child279)
     (waiting child279 table279)
     (waiting child104 table491)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child279)
     (served child104)
    )
  )
)
