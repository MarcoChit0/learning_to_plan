; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 684401

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child294 - child
    bread194 - bread-portion
    content246 - content-portion
    tray157 tray41 tray29 tray72 - tray
    table79 table479 table161 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray157 kitchen)
     (at tray41 kitchen)
     (at tray29 kitchen)
     (at tray72 kitchen)
     (at_kitchen_bread bread194)
     (at_kitchen_content content246)
     (not_allergic_gluten child294)
     (waiting child294 table161)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child294)
    )
  )
)
