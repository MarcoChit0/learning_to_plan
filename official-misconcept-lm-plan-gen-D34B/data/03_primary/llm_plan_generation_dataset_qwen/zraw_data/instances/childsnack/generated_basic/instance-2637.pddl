; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 885614

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child408 - child
    bread406 - bread-portion
    content114 - content-portion
    tray61 tray126 tray15 tray442 - tray
    table400 table198 table301 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray61 kitchen)
     (at tray126 kitchen)
     (at tray15 kitchen)
     (at tray442 kitchen)
     (at_kitchen_bread bread406)
     (at_kitchen_content content114)
     (not_allergic_gluten child408)
     (waiting child408 table198)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child408)
    )
  )
)
