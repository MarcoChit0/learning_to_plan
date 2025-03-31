; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 216786

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child20 - child
    bread317 - bread-portion
    content445 - content-portion
    tray39 - tray
    table439 table135 table4 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray39 kitchen)
     (at_kitchen_bread bread317)
     (at_kitchen_content content445)
     (not_allergic_gluten child20)
     (waiting child20 table4)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child20)
    )
  )
)
