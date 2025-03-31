; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 344629

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child221 child10 - child
    bread315 bread116 - bread-portion
    content393 content372 - content-portion
    tray482 tray129 tray223 - tray
    table461 table205 table68 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray482 kitchen)
     (at tray129 kitchen)
     (at tray223 kitchen)
     (at_kitchen_bread bread315)
     (at_kitchen_bread bread116)
     (at_kitchen_content content393)
     (at_kitchen_content content372)
     (no_gluten_bread bread315)
     (no_gluten_content content393)
     (allergic_gluten child221)
     (not_allergic_gluten child10)
     (waiting child221 table205)
     (waiting child10 table205)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child221)
     (served child10)
    )
  )
)
