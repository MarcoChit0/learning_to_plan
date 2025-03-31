; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 575677

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child305 - child
    bread66 - bread-portion
    content413 - content-portion
    tray94 tray39 tray356 tray3 - tray
    table217 table462 table95 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray94 kitchen)
     (at tray39 kitchen)
     (at tray356 kitchen)
     (at tray3 kitchen)
     (at_kitchen_bread bread66)
     (at_kitchen_content content413)
     (not_allergic_gluten child305)
     (waiting child305 table462)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child305)
    )
  )
)
