; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 410827

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child316 - child
    bread80 - bread-portion
    content72 - content-portion
    tray189 - tray
    table116 table360 table103 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray189 kitchen)
     (at_kitchen_bread bread80)
     (at_kitchen_content content72)
     (not_allergic_gluten child316)
     (waiting child316 table116)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child316)
    )
  )
)
