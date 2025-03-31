; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 679607

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child5 child217 - child
    bread169 bread184 - bread-portion
    content341 content93 - content-portion
    tray278 tray463 tray479 - tray
    table282 table347 table380 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray278 kitchen)
     (at tray463 kitchen)
     (at tray479 kitchen)
     (at_kitchen_bread bread169)
     (at_kitchen_bread bread184)
     (at_kitchen_content content341)
     (at_kitchen_content content93)
     (not_allergic_gluten child5)
     (not_allergic_gluten child217)
     (waiting child5 table347)
     (waiting child217 table380)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child5)
     (served child217)
    )
  )
)
