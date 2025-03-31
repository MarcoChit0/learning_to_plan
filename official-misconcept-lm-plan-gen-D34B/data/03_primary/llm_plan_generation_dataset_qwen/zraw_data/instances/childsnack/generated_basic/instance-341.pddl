; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 805320

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child31 - child
    bread141 - bread-portion
    content143 - content-portion
    tray53 tray234 tray17 tray99 - tray
    table353 table65 table175 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray53 kitchen)
     (at tray234 kitchen)
     (at tray17 kitchen)
     (at tray99 kitchen)
     (at_kitchen_bread bread141)
     (at_kitchen_content content143)
     (not_allergic_gluten child31)
     (waiting child31 table353)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child31)
    )
  )
)
