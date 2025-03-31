; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 588690

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child128 child108 - child
    bread195 bread116 - bread-portion
    content395 content385 - content-portion
    tray480 tray475 tray210 tray191 - tray
    table117 table18 table489 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray480 kitchen)
     (at tray475 kitchen)
     (at tray210 kitchen)
     (at tray191 kitchen)
     (at_kitchen_bread bread195)
     (at_kitchen_bread bread116)
     (at_kitchen_content content395)
     (at_kitchen_content content385)
     (not_allergic_gluten child128)
     (not_allergic_gluten child108)
     (waiting child128 table18)
     (waiting child108 table117)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child128)
     (served child108)
    )
  )
)
