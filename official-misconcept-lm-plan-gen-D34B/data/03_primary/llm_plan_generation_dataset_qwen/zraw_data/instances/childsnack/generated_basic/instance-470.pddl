; child-snack task with 3 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 706138

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child4 child343 child445 - child
    bread7 bread330 bread120 - bread-portion
    content319 content344 content408 - content-portion
    tray217 - tray
    table152 table330 table328 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray217 kitchen)
     (at_kitchen_bread bread7)
     (at_kitchen_bread bread330)
     (at_kitchen_bread bread120)
     (at_kitchen_content content319)
     (at_kitchen_content content344)
     (at_kitchen_content content408)
     (not_allergic_gluten child445)
     (not_allergic_gluten child4)
     (not_allergic_gluten child343)
     (waiting child4 table330)
     (waiting child343 table328)
     (waiting child445 table330)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child4)
     (served child343)
     (served child445)
    )
  )
)
