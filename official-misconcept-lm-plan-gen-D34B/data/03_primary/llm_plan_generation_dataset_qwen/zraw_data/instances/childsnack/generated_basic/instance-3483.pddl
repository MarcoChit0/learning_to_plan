; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 514612

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child365 child259 - child
    bread425 bread339 - bread-portion
    content39 content123 - content-portion
    tray324 tray69 - tray
    table462 table356 table230 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray324 kitchen)
     (at tray69 kitchen)
     (at_kitchen_bread bread425)
     (at_kitchen_bread bread339)
     (at_kitchen_content content39)
     (at_kitchen_content content123)
     (no_gluten_bread bread339)
     (no_gluten_content content39)
     (allergic_gluten child259)
     (not_allergic_gluten child365)
     (waiting child365 table356)
     (waiting child259 table462)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child365)
     (served child259)
    )
  )
)
