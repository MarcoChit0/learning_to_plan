; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 237323

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child308 child438 child273 - child
    bread445 bread106 bread123 - bread-portion
    content390 content453 content402 - content-portion
    tray139 tray411 tray382 - tray
    table39 table252 table60 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray139 kitchen)
     (at tray411 kitchen)
     (at tray382 kitchen)
     (at_kitchen_bread bread445)
     (at_kitchen_bread bread106)
     (at_kitchen_bread bread123)
     (at_kitchen_content content390)
     (at_kitchen_content content453)
     (at_kitchen_content content402)
     (not_allergic_gluten child308)
     (not_allergic_gluten child273)
     (not_allergic_gluten child438)
     (waiting child308 table39)
     (waiting child438 table252)
     (waiting child273 table60)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child308)
     (served child438)
     (served child273)
    )
  )
)
