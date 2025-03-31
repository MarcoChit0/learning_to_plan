; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 306607

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child423 child325 - child
    bread114 bread137 - bread-portion
    content415 content488 - content-portion
    tray98 tray129 tray22 - tray
    table340 table144 table388 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray98 kitchen)
     (at tray129 kitchen)
     (at tray22 kitchen)
     (at_kitchen_bread bread114)
     (at_kitchen_bread bread137)
     (at_kitchen_content content415)
     (at_kitchen_content content488)
     (not_allergic_gluten child423)
     (not_allergic_gluten child325)
     (waiting child423 table388)
     (waiting child325 table340)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child423)
     (served child325)
    )
  )
)
