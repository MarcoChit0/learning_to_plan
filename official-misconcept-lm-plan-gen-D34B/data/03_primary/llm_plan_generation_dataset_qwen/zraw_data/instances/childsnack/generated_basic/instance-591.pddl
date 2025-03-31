; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 770994

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child462 - child
    bread103 - bread-portion
    content38 - content-portion
    tray106 - tray
    table143 table101 table3 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray106 kitchen)
     (at_kitchen_bread bread103)
     (at_kitchen_content content38)
     (not_allergic_gluten child462)
     (waiting child462 table3)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child462)
    )
  )
)
