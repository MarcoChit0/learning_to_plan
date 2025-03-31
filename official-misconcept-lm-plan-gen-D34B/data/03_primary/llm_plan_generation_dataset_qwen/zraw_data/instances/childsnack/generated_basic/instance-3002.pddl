; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 170836

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child70 child182 - child
    bread297 bread84 - bread-portion
    content275 content450 - content-portion
    tray69 tray489 tray230 tray120 - tray
    table107 table40 table109 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray69 kitchen)
     (at tray489 kitchen)
     (at tray230 kitchen)
     (at tray120 kitchen)
     (at_kitchen_bread bread297)
     (at_kitchen_bread bread84)
     (at_kitchen_content content275)
     (at_kitchen_content content450)
     (not_allergic_gluten child182)
     (not_allergic_gluten child70)
     (waiting child70 table107)
     (waiting child182 table109)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child70)
     (served child182)
    )
  )
)
