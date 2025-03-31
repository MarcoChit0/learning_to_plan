; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 478555

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child450 child279 - child
    bread372 bread430 - bread-portion
    content445 content359 - content-portion
    tray294 tray119 - tray
    table151 table397 table109 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray294 kitchen)
     (at tray119 kitchen)
     (at_kitchen_bread bread372)
     (at_kitchen_bread bread430)
     (at_kitchen_content content445)
     (at_kitchen_content content359)
     (no_gluten_bread bread430)
     (no_gluten_content content445)
     (allergic_gluten child450)
     (not_allergic_gluten child279)
     (waiting child450 table151)
     (waiting child279 table151)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child450)
     (served child279)
    )
  )
)
