; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 590119

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child269 child169 - child
    bread376 bread172 - bread-portion
    content181 content73 - content-portion
    tray318 tray405 - tray
    table78 table103 table428 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray318 kitchen)
     (at tray405 kitchen)
     (at_kitchen_bread bread376)
     (at_kitchen_bread bread172)
     (at_kitchen_content content181)
     (at_kitchen_content content73)
     (not_allergic_gluten child169)
     (not_allergic_gluten child269)
     (waiting child269 table78)
     (waiting child169 table428)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child269)
     (served child169)
    )
  )
)
