; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 314069

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child354 - child
    bread12 - bread-portion
    content21 - content-portion
    tray95 tray197 tray224 tray337 - tray
    table203 table142 table180 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray95 kitchen)
     (at tray197 kitchen)
     (at tray224 kitchen)
     (at tray337 kitchen)
     (at_kitchen_bread bread12)
     (at_kitchen_content content21)
     (not_allergic_gluten child354)
     (waiting child354 table203)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child354)
    )
  )
)
