; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 349309

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child113 child275 - child
    bread452 bread210 - bread-portion
    content428 content205 - content-portion
    tray8 tray30 tray248 - tray
    table234 table32 table378 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray8 kitchen)
     (at tray30 kitchen)
     (at tray248 kitchen)
     (at_kitchen_bread bread452)
     (at_kitchen_bread bread210)
     (at_kitchen_content content428)
     (at_kitchen_content content205)
     (no_gluten_bread bread452)
     (no_gluten_content content428)
     (allergic_gluten child113)
     (not_allergic_gluten child275)
     (waiting child113 table234)
     (waiting child275 table378)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child113)
     (served child275)
    )
  )
)
