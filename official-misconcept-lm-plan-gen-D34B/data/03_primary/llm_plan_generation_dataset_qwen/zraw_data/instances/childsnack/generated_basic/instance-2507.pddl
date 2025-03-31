; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 819245

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child387 - child
    bread472 - bread-portion
    content58 - content-portion
    tray51 tray89 tray135 - tray
    table70 table495 table257 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray51 kitchen)
     (at tray89 kitchen)
     (at tray135 kitchen)
     (at_kitchen_bread bread472)
     (at_kitchen_content content58)
     (not_allergic_gluten child387)
     (waiting child387 table495)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child387)
    )
  )
)
