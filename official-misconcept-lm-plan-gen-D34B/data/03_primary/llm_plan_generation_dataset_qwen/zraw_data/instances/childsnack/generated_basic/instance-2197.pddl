; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 681002

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child497 - child
    bread466 - bread-portion
    content365 - content-portion
    tray445 - tray
    table243 table109 table318 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray445 kitchen)
     (at_kitchen_bread bread466)
     (at_kitchen_content content365)
     (not_allergic_gluten child497)
     (waiting child497 table318)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child497)
    )
  )
)
