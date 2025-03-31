; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 157062

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child354 - child
    bread367 - bread-portion
    content262 - content-portion
    tray202 - tray
    table474 table197 table403 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray202 kitchen)
     (at_kitchen_bread bread367)
     (at_kitchen_content content262)
     (not_allergic_gluten child354)
     (waiting child354 table403)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child354)
    )
  )
)
