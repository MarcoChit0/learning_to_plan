; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 50625

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child156 - child
    bread312 - bread-portion
    content119 - content-portion
    tray209 - tray
    table253 table58 table299 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray209 kitchen)
     (at_kitchen_bread bread312)
     (at_kitchen_content content119)
     (not_allergic_gluten child156)
     (waiting child156 table253)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child156)
    )
  )
)
