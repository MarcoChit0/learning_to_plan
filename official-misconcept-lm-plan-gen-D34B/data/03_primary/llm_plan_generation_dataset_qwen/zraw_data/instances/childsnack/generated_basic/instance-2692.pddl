; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 157308

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child55 child238 child484 - child
    bread445 bread400 bread457 - bread-portion
    content459 content487 content156 - content-portion
    tray188 tray158 tray297 - tray
    table498 table133 table164 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray188 kitchen)
     (at tray158 kitchen)
     (at tray297 kitchen)
     (at_kitchen_bread bread445)
     (at_kitchen_bread bread400)
     (at_kitchen_bread bread457)
     (at_kitchen_content content459)
     (at_kitchen_content content487)
     (at_kitchen_content content156)
     (not_allergic_gluten child55)
     (not_allergic_gluten child484)
     (not_allergic_gluten child238)
     (waiting child55 table133)
     (waiting child238 table498)
     (waiting child484 table498)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child55)
     (served child238)
     (served child484)
    )
  )
)
