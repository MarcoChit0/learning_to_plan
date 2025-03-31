; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 639448

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child346 - child
    bread296 - bread-portion
    content151 - content-portion
    tray331 tray445 tray216 tray464 - tray
    table16 table310 table271 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray331 kitchen)
     (at tray445 kitchen)
     (at tray216 kitchen)
     (at tray464 kitchen)
     (at_kitchen_bread bread296)
     (at_kitchen_content content151)
     (not_allergic_gluten child346)
     (waiting child346 table16)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child346)
    )
  )
)
