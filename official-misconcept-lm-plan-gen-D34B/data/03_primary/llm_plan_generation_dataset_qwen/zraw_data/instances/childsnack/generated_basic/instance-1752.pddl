; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 732535

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child444 - child
    bread354 - bread-portion
    content282 - content-portion
    tray202 tray473 tray347 - tray
    table403 table290 table368 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray202 kitchen)
     (at tray473 kitchen)
     (at tray347 kitchen)
     (at_kitchen_bread bread354)
     (at_kitchen_content content282)
     (not_allergic_gluten child444)
     (waiting child444 table403)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child444)
    )
  )
)
