; child-snack task with 2 children and 0.51 gluten factor 
; constant factor of 1.9
; random seed: 496509

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child220 child270 - child
    bread335 bread445 - bread-portion
    content329 content225 - content-portion
    tray226 tray247 tray399 - tray
    table385 table368 table478 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray226 kitchen)
     (at tray247 kitchen)
     (at tray399 kitchen)
     (at_kitchen_bread bread335)
     (at_kitchen_bread bread445)
     (at_kitchen_content content329)
     (at_kitchen_content content225)
     (no_gluten_bread bread445)
     (no_gluten_content content225)
     (allergic_gluten child270)
     (not_allergic_gluten child220)
     (waiting child220 table385)
     (waiting child270 table478)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child220)
     (served child270)
    )
  )
)
