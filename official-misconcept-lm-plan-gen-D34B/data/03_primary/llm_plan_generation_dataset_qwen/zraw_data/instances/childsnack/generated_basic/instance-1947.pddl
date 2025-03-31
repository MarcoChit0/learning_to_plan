; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 407651

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child98 - child
    bread66 - bread-portion
    content319 - content-portion
    tray219 tray424 tray220 - tray
    table179 table34 table233 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray219 kitchen)
     (at tray424 kitchen)
     (at tray220 kitchen)
     (at_kitchen_bread bread66)
     (at_kitchen_content content319)
     (not_allergic_gluten child98)
     (waiting child98 table34)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child98)
    )
  )
)
