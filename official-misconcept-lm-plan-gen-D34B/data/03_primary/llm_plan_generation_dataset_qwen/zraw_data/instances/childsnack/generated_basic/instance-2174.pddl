; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 681029

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child477 - child
    bread365 - bread-portion
    content359 - content-portion
    tray429 - tray
    table439 table90 table146 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray429 kitchen)
     (at_kitchen_bread bread365)
     (at_kitchen_content content359)
     (not_allergic_gluten child477)
     (waiting child477 table439)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child477)
    )
  )
)
