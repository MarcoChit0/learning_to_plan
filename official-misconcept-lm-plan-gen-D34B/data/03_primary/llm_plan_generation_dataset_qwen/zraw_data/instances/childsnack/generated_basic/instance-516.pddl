; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 238436

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child387 child388 - child
    bread148 bread352 - bread-portion
    content257 content465 - content-portion
    tray42 tray304 - tray
    table42 table371 table282 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray42 kitchen)
     (at tray304 kitchen)
     (at_kitchen_bread bread148)
     (at_kitchen_bread bread352)
     (at_kitchen_content content257)
     (at_kitchen_content content465)
     (not_allergic_gluten child387)
     (not_allergic_gluten child388)
     (waiting child387 table282)
     (waiting child388 table371)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child387)
     (served child388)
    )
  )
)
