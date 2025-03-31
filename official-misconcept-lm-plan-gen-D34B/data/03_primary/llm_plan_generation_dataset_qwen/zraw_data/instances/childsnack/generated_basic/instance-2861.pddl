; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 272141

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child320 - child
    bread116 - bread-portion
    content450 - content-portion
    tray332 tray275 - tray
    table210 table427 table433 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray332 kitchen)
     (at tray275 kitchen)
     (at_kitchen_bread bread116)
     (at_kitchen_content content450)
     (not_allergic_gluten child320)
     (waiting child320 table433)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child320)
    )
  )
)
