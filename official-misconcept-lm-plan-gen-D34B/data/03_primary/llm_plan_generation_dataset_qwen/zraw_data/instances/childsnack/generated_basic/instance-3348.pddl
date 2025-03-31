; child-snack task with 2 children and 0.6 gluten factor 
; constant factor of 1.3
; random seed: 553280

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child202 child11 - child
    bread321 bread126 - bread-portion
    content48 content445 - content-portion
    tray454 tray337 - tray
    table366 table490 table424 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray454 kitchen)
     (at tray337 kitchen)
     (at_kitchen_bread bread321)
     (at_kitchen_bread bread126)
     (at_kitchen_content content48)
     (at_kitchen_content content445)
     (no_gluten_bread bread126)
     (no_gluten_content content445)
     (allergic_gluten child11)
     (not_allergic_gluten child202)
     (waiting child202 table424)
     (waiting child11 table366)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child202)
     (served child11)
    )
  )
)
