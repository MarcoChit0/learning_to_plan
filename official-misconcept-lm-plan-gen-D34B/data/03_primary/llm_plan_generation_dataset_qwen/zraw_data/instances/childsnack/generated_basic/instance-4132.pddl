; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 288448

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child138 child494 - child
    bread117 bread484 - bread-portion
    content7 content450 - content-portion
    tray333 tray360 tray214 - tray
    table171 table181 table465 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray333 kitchen)
     (at tray360 kitchen)
     (at tray214 kitchen)
     (at_kitchen_bread bread117)
     (at_kitchen_bread bread484)
     (at_kitchen_content content7)
     (at_kitchen_content content450)
     (not_allergic_gluten child138)
     (not_allergic_gluten child494)
     (waiting child138 table181)
     (waiting child494 table465)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child138)
     (served child494)
    )
  )
)
