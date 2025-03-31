; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 11349

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child54 child370 - child
    bread236 bread494 - bread-portion
    content478 content444 - content-portion
    tray62 tray465 - tray
    table259 table193 table125 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray62 kitchen)
     (at tray465 kitchen)
     (at_kitchen_bread bread236)
     (at_kitchen_bread bread494)
     (at_kitchen_content content478)
     (at_kitchen_content content444)
     (no_gluten_bread bread236)
     (no_gluten_content content444)
     (allergic_gluten child54)
     (not_allergic_gluten child370)
     (waiting child54 table259)
     (waiting child370 table193)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child54)
     (served child370)
    )
  )
)
