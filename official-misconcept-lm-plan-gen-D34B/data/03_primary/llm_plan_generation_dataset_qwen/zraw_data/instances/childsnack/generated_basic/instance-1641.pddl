; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 882273

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child405 - child
    bread298 - bread-portion
    content179 - content-portion
    tray273 tray222 tray382 tray340 - tray
    table108 table152 table120 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray273 kitchen)
     (at tray222 kitchen)
     (at tray382 kitchen)
     (at tray340 kitchen)
     (at_kitchen_bread bread298)
     (at_kitchen_content content179)
     (not_allergic_gluten child405)
     (waiting child405 table120)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child405)
    )
  )
)
