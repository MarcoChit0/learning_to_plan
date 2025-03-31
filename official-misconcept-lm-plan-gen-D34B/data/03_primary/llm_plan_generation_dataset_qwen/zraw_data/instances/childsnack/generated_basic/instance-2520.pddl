; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 186471

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child452 - child
    bread54 - bread-portion
    content498 - content-portion
    tray286 tray193 tray144 - tray
    table157 table63 table10 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray286 kitchen)
     (at tray193 kitchen)
     (at tray144 kitchen)
     (at_kitchen_bread bread54)
     (at_kitchen_content content498)
     (not_allergic_gluten child452)
     (waiting child452 table157)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child452)
    )
  )
)
