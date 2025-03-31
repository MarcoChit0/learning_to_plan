; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 32578

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child494 - child
    bread232 - bread-portion
    content396 - content-portion
    tray183 tray67 tray308 tray382 - tray
    table194 table440 table182 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray183 kitchen)
     (at tray67 kitchen)
     (at tray308 kitchen)
     (at tray382 kitchen)
     (at_kitchen_bread bread232)
     (at_kitchen_content content396)
     (not_allergic_gluten child494)
     (waiting child494 table194)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child494)
    )
  )
)
