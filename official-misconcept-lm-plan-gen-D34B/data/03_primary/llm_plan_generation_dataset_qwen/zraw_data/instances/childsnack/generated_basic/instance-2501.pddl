; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 868565

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child83 child8 - child
    bread36 bread397 - bread-portion
    content465 content80 - content-portion
    tray367 tray445 - tray
    table498 table67 table483 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray367 kitchen)
     (at tray445 kitchen)
     (at_kitchen_bread bread36)
     (at_kitchen_bread bread397)
     (at_kitchen_content content465)
     (at_kitchen_content content80)
     (not_allergic_gluten child83)
     (not_allergic_gluten child8)
     (waiting child83 table498)
     (waiting child8 table498)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child83)
     (served child8)
    )
  )
)
