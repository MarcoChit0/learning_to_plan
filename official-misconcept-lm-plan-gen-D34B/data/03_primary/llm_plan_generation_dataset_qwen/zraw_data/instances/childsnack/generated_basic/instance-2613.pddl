; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 443952

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child215 - child
    bread365 - bread-portion
    content160 - content-portion
    tray160 tray37 tray485 - tray
    table47 table383 table316 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray160 kitchen)
     (at tray37 kitchen)
     (at tray485 kitchen)
     (at_kitchen_bread bread365)
     (at_kitchen_content content160)
     (not_allergic_gluten child215)
     (waiting child215 table47)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child215)
    )
  )
)
