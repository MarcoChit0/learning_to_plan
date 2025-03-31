; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 843071

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child271 child390 - child
    bread155 bread399 - bread-portion
    content366 content344 - content-portion
    tray414 tray320 - tray
    table228 table445 table322 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray414 kitchen)
     (at tray320 kitchen)
     (at_kitchen_bread bread155)
     (at_kitchen_bread bread399)
     (at_kitchen_content content366)
     (at_kitchen_content content344)
     (not_allergic_gluten child271)
     (not_allergic_gluten child390)
     (waiting child271 table228)
     (waiting child390 table445)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child271)
     (served child390)
    )
  )
)
