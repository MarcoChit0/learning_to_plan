; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 126631

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child432 - child
    bread2 - bread-portion
    content13 - content-portion
    tray51 tray100 - tray
    table336 table425 table400 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray51 kitchen)
     (at tray100 kitchen)
     (at_kitchen_bread bread2)
     (at_kitchen_content content13)
     (not_allergic_gluten child432)
     (waiting child432 table425)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child432)
    )
  )
)
