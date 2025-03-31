; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 212191

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child354 - child
    bread393 - bread-portion
    content57 - content-portion
    tray170 tray175 - tray
    table357 table170 table416 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray170 kitchen)
     (at tray175 kitchen)
     (at_kitchen_bread bread393)
     (at_kitchen_content content57)
     (not_allergic_gluten child354)
     (waiting child354 table357)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child354)
    )
  )
)
