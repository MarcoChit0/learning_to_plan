; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 872588

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child289 - child
    bread249 - bread-portion
    content468 - content-portion
    tray462 - tray
    table166 table69 table401 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray462 kitchen)
     (at_kitchen_bread bread249)
     (at_kitchen_content content468)
     (not_allergic_gluten child289)
     (waiting child289 table401)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child289)
    )
  )
)
