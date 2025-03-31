; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 149796

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child8 - child
    bread381 - bread-portion
    content376 - content-portion
    tray108 tray160 - tray
    table31 table284 table340 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray108 kitchen)
     (at tray160 kitchen)
     (at_kitchen_bread bread381)
     (at_kitchen_content content376)
     (not_allergic_gluten child8)
     (waiting child8 table284)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child8)
    )
  )
)
