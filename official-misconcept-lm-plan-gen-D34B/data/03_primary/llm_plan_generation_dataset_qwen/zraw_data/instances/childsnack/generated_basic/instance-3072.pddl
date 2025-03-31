; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 613170

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child54 child443 - child
    bread402 bread74 - bread-portion
    content411 content82 - content-portion
    tray207 tray72 tray178 - tray
    table445 table402 table214 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray207 kitchen)
     (at tray72 kitchen)
     (at tray178 kitchen)
     (at_kitchen_bread bread402)
     (at_kitchen_bread bread74)
     (at_kitchen_content content411)
     (at_kitchen_content content82)
     (no_gluten_bread bread74)
     (no_gluten_content content411)
     (allergic_gluten child54)
     (not_allergic_gluten child443)
     (waiting child54 table445)
     (waiting child443 table402)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child54)
     (served child443)
    )
  )
)
