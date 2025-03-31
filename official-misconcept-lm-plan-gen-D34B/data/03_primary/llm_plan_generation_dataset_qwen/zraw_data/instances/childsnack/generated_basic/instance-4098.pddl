; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 390086

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child14 - child
    bread154 - bread-portion
    content300 - content-portion
    tray292 tray48 tray178 - tray
    table435 table72 table182 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray292 kitchen)
     (at tray48 kitchen)
     (at tray178 kitchen)
     (at_kitchen_bread bread154)
     (at_kitchen_content content300)
     (not_allergic_gluten child14)
     (waiting child14 table72)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child14)
    )
  )
)
