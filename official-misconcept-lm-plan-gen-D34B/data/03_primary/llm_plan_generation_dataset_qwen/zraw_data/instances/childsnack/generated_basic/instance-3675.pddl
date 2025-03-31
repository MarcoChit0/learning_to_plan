; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 687195

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child431 child483 - child
    bread247 bread76 - bread-portion
    content478 content11 - content-portion
    tray147 tray387 - tray
    table373 table490 table353 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray147 kitchen)
     (at tray387 kitchen)
     (at_kitchen_bread bread247)
     (at_kitchen_bread bread76)
     (at_kitchen_content content478)
     (at_kitchen_content content11)
     (no_gluten_bread bread76)
     (no_gluten_content content11)
     (allergic_gluten child483)
     (not_allergic_gluten child431)
     (waiting child431 table490)
     (waiting child483 table490)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child431)
     (served child483)
    )
  )
)
