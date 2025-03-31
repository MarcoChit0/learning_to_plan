; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 403419

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child438 child235 child311 - child
    bread430 bread233 bread50 - bread-portion
    content12 content160 content141 - content-portion
    tray394 - tray
    table445 table230 table186 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray394 kitchen)
     (at_kitchen_bread bread430)
     (at_kitchen_bread bread233)
     (at_kitchen_bread bread50)
     (at_kitchen_content content12)
     (at_kitchen_content content160)
     (at_kitchen_content content141)
     (not_allergic_gluten child311)
     (not_allergic_gluten child235)
     (not_allergic_gluten child438)
     (waiting child438 table445)
     (waiting child235 table445)
     (waiting child311 table230)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child438)
     (served child235)
     (served child311)
    )
  )
)
