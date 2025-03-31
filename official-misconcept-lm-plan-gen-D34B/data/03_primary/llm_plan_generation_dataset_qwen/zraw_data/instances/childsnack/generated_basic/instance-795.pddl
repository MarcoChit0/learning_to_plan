; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 872910

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child59 - child
    bread242 - bread-portion
    content385 - content-portion
    tray376 tray3 - tray
    table272 table415 table298 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray376 kitchen)
     (at tray3 kitchen)
     (at_kitchen_bread bread242)
     (at_kitchen_content content385)
     (not_allergic_gluten child59)
     (waiting child59 table415)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child59)
    )
  )
)
