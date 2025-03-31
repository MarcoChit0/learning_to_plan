; child-snack task with 2 children and 0.45 gluten factor 
; constant factor of 1.7
; random seed: 399856

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child413 child268 - child
    bread112 bread130 - bread-portion
    content4 content89 - content-portion
    tray499 tray109 tray354 - tray
    table438 table72 table186 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray499 kitchen)
     (at tray109 kitchen)
     (at tray354 kitchen)
     (at_kitchen_bread bread112)
     (at_kitchen_bread bread130)
     (at_kitchen_content content4)
     (at_kitchen_content content89)
     (not_allergic_gluten child268)
     (not_allergic_gluten child413)
     (waiting child413 table438)
     (waiting child268 table438)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child413)
     (served child268)
    )
  )
)
