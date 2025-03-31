; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 437440

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child246 child81 - child
    bread352 bread311 - bread-portion
    content76 content198 - content-portion
    tray36 tray482 tray68 - tray
    table261 table336 table244 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray36 kitchen)
     (at tray482 kitchen)
     (at tray68 kitchen)
     (at_kitchen_bread bread352)
     (at_kitchen_bread bread311)
     (at_kitchen_content content76)
     (at_kitchen_content content198)
     (not_allergic_gluten child81)
     (not_allergic_gluten child246)
     (waiting child246 table261)
     (waiting child81 table244)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child246)
     (served child81)
    )
  )
)
