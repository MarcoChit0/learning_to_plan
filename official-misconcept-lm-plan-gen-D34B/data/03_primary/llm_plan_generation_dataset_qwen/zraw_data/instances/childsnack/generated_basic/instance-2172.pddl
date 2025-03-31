; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 707241

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child293 - child
    bread498 - bread-portion
    content405 - content-portion
    tray236 - tray
    table347 table310 table166 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray236 kitchen)
     (at_kitchen_bread bread498)
     (at_kitchen_content content405)
     (not_allergic_gluten child293)
     (waiting child293 table166)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child293)
    )
  )
)
