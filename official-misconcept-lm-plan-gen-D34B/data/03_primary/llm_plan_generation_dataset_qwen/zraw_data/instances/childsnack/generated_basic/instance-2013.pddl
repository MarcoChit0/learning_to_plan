; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 718536

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child128 child292 - child
    bread314 bread477 - bread-portion
    content436 content347 - content-portion
    tray360 tray457 tray328 - tray
    table364 table127 table490 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray360 kitchen)
     (at tray457 kitchen)
     (at tray328 kitchen)
     (at_kitchen_bread bread314)
     (at_kitchen_bread bread477)
     (at_kitchen_content content436)
     (at_kitchen_content content347)
     (not_allergic_gluten child128)
     (not_allergic_gluten child292)
     (waiting child128 table127)
     (waiting child292 table364)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child128)
     (served child292)
    )
  )
)
