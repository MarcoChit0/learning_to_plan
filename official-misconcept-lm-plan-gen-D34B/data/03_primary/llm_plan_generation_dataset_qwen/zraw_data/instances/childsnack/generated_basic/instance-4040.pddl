; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 225541

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child128 child43 child263 - child
    bread420 bread445 bread374 - bread-portion
    content472 content124 content39 - content-portion
    tray108 - tray
    table108 table228 table200 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray108 kitchen)
     (at_kitchen_bread bread420)
     (at_kitchen_bread bread445)
     (at_kitchen_bread bread374)
     (at_kitchen_content content472)
     (at_kitchen_content content124)
     (at_kitchen_content content39)
     (not_allergic_gluten child263)
     (not_allergic_gluten child128)
     (not_allergic_gluten child43)
     (waiting child128 table228)
     (waiting child43 table228)
     (waiting child263 table108)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child128)
     (served child43)
     (served child263)
    )
  )
)
