; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 170695

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child492 child3 - child
    bread488 bread124 - bread-portion
    content442 content267 - content-portion
    tray315 tray244 - tray
    table354 table334 table19 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray315 kitchen)
     (at tray244 kitchen)
     (at_kitchen_bread bread488)
     (at_kitchen_bread bread124)
     (at_kitchen_content content442)
     (at_kitchen_content content267)
     (not_allergic_gluten child3)
     (not_allergic_gluten child492)
     (waiting child492 table354)
     (waiting child3 table19)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child492)
     (served child3)
    )
  )
)
