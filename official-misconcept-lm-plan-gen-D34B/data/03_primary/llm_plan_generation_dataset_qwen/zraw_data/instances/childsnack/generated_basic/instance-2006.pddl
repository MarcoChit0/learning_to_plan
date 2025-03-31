; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 113136

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child188 child354 - child
    bread237 bread169 - bread-portion
    content379 content330 - content-portion
    tray25 tray418 tray227 - tray
    table134 table171 table370 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray25 kitchen)
     (at tray418 kitchen)
     (at tray227 kitchen)
     (at_kitchen_bread bread237)
     (at_kitchen_bread bread169)
     (at_kitchen_content content379)
     (at_kitchen_content content330)
     (not_allergic_gluten child188)
     (not_allergic_gluten child354)
     (waiting child188 table171)
     (waiting child354 table370)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child188)
     (served child354)
    )
  )
)
