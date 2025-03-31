; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 345501

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child33 - child
    bread304 - bread-portion
    content205 - content-portion
    tray121 tray475 - tray
    table154 table408 table21 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray121 kitchen)
     (at tray475 kitchen)
     (at_kitchen_bread bread304)
     (at_kitchen_content content205)
     (not_allergic_gluten child33)
     (waiting child33 table154)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child33)
    )
  )
)
