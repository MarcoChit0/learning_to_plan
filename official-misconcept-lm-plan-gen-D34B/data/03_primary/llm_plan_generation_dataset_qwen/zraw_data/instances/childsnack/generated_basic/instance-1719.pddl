; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 263080

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child181 - child
    bread393 - bread-portion
    content400 - content-portion
    tray459 tray358 tray356 - tray
    table3 table13 table437 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray459 kitchen)
     (at tray358 kitchen)
     (at tray356 kitchen)
     (at_kitchen_bread bread393)
     (at_kitchen_content content400)
     (not_allergic_gluten child181)
     (waiting child181 table3)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child181)
    )
  )
)
