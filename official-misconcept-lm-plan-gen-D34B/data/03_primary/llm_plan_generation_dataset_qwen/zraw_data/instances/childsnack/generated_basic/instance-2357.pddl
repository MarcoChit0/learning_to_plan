; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 919272

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child240 child428 - child
    bread433 bread499 - bread-portion
    content298 content194 - content-portion
    tray111 - tray
    table232 table495 table101 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray111 kitchen)
     (at_kitchen_bread bread433)
     (at_kitchen_bread bread499)
     (at_kitchen_content content298)
     (at_kitchen_content content194)
     (not_allergic_gluten child428)
     (not_allergic_gluten child240)
     (waiting child240 table101)
     (waiting child428 table232)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child240)
     (served child428)
    )
  )
)
