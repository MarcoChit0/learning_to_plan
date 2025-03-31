; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 524999

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child87 - child
    bread107 - bread-portion
    content143 - content-portion
    tray354 - tray
    table400 table90 table340 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray354 kitchen)
     (at_kitchen_bread bread107)
     (at_kitchen_content content143)
     (not_allergic_gluten child87)
     (waiting child87 table400)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child87)
    )
  )
)
