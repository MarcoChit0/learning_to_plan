; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 342847

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child193 - child
    bread190 - bread-portion
    content46 - content-portion
    tray297 tray445 tray75 - tray
    table83 table95 table52 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray297 kitchen)
     (at tray445 kitchen)
     (at tray75 kitchen)
     (at_kitchen_bread bread190)
     (at_kitchen_content content46)
     (not_allergic_gluten child193)
     (waiting child193 table52)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child193)
    )
  )
)
