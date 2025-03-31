; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 648174

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child306 - child
    bread134 - bread-portion
    content157 - content-portion
    tray283 tray113 tray292 - tray
    table114 table384 table93 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray283 kitchen)
     (at tray113 kitchen)
     (at tray292 kitchen)
     (at_kitchen_bread bread134)
     (at_kitchen_content content157)
     (not_allergic_gluten child306)
     (waiting child306 table384)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child306)
    )
  )
)
