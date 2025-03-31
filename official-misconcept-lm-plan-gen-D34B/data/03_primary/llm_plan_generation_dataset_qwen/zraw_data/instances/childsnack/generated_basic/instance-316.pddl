; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 161488

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child354 - child
    bread300 - bread-portion
    content129 - content-portion
    tray194 - tray
    table385 table56 table227 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray194 kitchen)
     (at_kitchen_bread bread300)
     (at_kitchen_content content129)
     (not_allergic_gluten child354)
     (waiting child354 table227)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child354)
    )
  )
)
