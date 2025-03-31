; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 905143

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child146 child467 - child
    bread248 bread270 - bread-portion
    content368 content39 - content-portion
    tray333 tray469 - tray
    table274 table441 table332 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray333 kitchen)
     (at tray469 kitchen)
     (at_kitchen_bread bread248)
     (at_kitchen_bread bread270)
     (at_kitchen_content content368)
     (at_kitchen_content content39)
     (not_allergic_gluten child467)
     (not_allergic_gluten child146)
     (waiting child146 table441)
     (waiting child467 table274)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child146)
     (served child467)
    )
  )
)
