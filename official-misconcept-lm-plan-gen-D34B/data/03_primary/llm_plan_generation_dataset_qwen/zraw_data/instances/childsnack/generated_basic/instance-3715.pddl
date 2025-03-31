; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 388539

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child7 - child
    bread116 - bread-portion
    content240 - content-portion
    tray200 tray61 tray203 - tray
    table401 table466 table275 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray200 kitchen)
     (at tray61 kitchen)
     (at tray203 kitchen)
     (at_kitchen_bread bread116)
     (at_kitchen_content content240)
     (not_allergic_gluten child7)
     (waiting child7 table275)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child7)
    )
  )
)
