; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 262255

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child431 child12 - child
    bread250 bread42 - bread-portion
    content90 content109 - content-portion
    tray286 tray279 tray18 tray498 - tray
    table286 table306 table435 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray286 kitchen)
     (at tray279 kitchen)
     (at tray18 kitchen)
     (at tray498 kitchen)
     (at_kitchen_bread bread250)
     (at_kitchen_bread bread42)
     (at_kitchen_content content90)
     (at_kitchen_content content109)
     (not_allergic_gluten child431)
     (not_allergic_gluten child12)
     (waiting child431 table306)
     (waiting child12 table306)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child431)
     (served child12)
    )
  )
)
