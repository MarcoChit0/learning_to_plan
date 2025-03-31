; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 235183

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child242 - child
    bread320 - bread-portion
    content442 - content-portion
    tray11 tray125 tray181 tray52 - tray
    table446 table67 table344 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray11 kitchen)
     (at tray125 kitchen)
     (at tray181 kitchen)
     (at tray52 kitchen)
     (at_kitchen_bread bread320)
     (at_kitchen_content content442)
     (not_allergic_gluten child242)
     (waiting child242 table67)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child242)
    )
  )
)
