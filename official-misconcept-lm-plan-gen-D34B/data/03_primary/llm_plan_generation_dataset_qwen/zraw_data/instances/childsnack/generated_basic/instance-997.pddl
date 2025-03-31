; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 198593

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child206 child197 child485 - child
    bread339 bread122 bread21 - bread-portion
    content424 content445 content331 - content-portion
    tray30 - tray
    table431 table30 table112 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray30 kitchen)
     (at_kitchen_bread bread339)
     (at_kitchen_bread bread122)
     (at_kitchen_bread bread21)
     (at_kitchen_content content424)
     (at_kitchen_content content445)
     (at_kitchen_content content331)
     (not_allergic_gluten child485)
     (not_allergic_gluten child206)
     (not_allergic_gluten child197)
     (waiting child206 table431)
     (waiting child197 table112)
     (waiting child485 table30)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child206)
     (served child197)
     (served child485)
    )
  )
)
