; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 907655

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child35 - child
    bread309 - bread-portion
    content438 - content-portion
    tray116 tray136 - tray
    table288 table442 table240 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray116 kitchen)
     (at tray136 kitchen)
     (at_kitchen_bread bread309)
     (at_kitchen_content content438)
     (not_allergic_gluten child35)
     (waiting child35 table240)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child35)
    )
  )
)
