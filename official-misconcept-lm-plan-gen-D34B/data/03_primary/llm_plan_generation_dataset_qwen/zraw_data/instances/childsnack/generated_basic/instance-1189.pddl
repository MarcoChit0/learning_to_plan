; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 831672

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child334 child445 child362 - child
    bread88 bread475 bread147 - bread-portion
    content55 content296 content23 - content-portion
    tray217 tray472 tray349 - tray
    table325 table191 table274 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray217 kitchen)
     (at tray472 kitchen)
     (at tray349 kitchen)
     (at_kitchen_bread bread88)
     (at_kitchen_bread bread475)
     (at_kitchen_bread bread147)
     (at_kitchen_content content55)
     (at_kitchen_content content296)
     (at_kitchen_content content23)
     (not_allergic_gluten child362)
     (not_allergic_gluten child334)
     (not_allergic_gluten child445)
     (waiting child334 table191)
     (waiting child445 table325)
     (waiting child362 table274)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child334)
     (served child445)
     (served child362)
    )
  )
)
