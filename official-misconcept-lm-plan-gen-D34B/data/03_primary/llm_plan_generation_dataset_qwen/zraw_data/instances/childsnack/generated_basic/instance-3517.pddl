; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 621981

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child180 child115 - child
    bread323 bread360 - bread-portion
    content283 content193 - content-portion
    tray253 tray218 - tray
    table332 table27 table407 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray253 kitchen)
     (at tray218 kitchen)
     (at_kitchen_bread bread323)
     (at_kitchen_bread bread360)
     (at_kitchen_content content283)
     (at_kitchen_content content193)
     (not_allergic_gluten child115)
     (not_allergic_gluten child180)
     (waiting child180 table407)
     (waiting child115 table27)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child180)
     (served child115)
    )
  )
)
