; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 113142

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child413 child137 - child
    bread425 bread120 - bread-portion
    content101 content65 - content-portion
    tray458 tray203 - tray
    table393 table19 table78 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray458 kitchen)
     (at tray203 kitchen)
     (at_kitchen_bread bread425)
     (at_kitchen_bread bread120)
     (at_kitchen_content content101)
     (at_kitchen_content content65)
     (not_allergic_gluten child413)
     (not_allergic_gluten child137)
     (waiting child413 table78)
     (waiting child137 table19)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child413)
     (served child137)
    )
  )
)
