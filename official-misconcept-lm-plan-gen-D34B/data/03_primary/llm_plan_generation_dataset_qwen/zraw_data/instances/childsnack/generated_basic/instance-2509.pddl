; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 542661

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child452 - child
    bread481 - bread-portion
    content479 - content-portion
    tray119 tray152 tray323 - tray
    table35 table486 table39 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray119 kitchen)
     (at tray152 kitchen)
     (at tray323 kitchen)
     (at_kitchen_bread bread481)
     (at_kitchen_content content479)
     (not_allergic_gluten child452)
     (waiting child452 table35)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child452)
    )
  )
)
