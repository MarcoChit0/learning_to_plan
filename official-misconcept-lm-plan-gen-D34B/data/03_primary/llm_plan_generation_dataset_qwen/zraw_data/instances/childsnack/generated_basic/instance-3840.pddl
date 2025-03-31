; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 861117

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child143 - child
    bread92 - bread-portion
    content192 - content-portion
    tray166 tray156 tray121 tray284 - tray
    table212 table279 table8 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray166 kitchen)
     (at tray156 kitchen)
     (at tray121 kitchen)
     (at tray284 kitchen)
     (at_kitchen_bread bread92)
     (at_kitchen_content content192)
     (not_allergic_gluten child143)
     (waiting child143 table8)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child143)
    )
  )
)
