; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 519016

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child45 child297 - child
    bread42 bread453 - bread-portion
    content176 content68 - content-portion
    tray113 tray224 tray420 - tray
    table169 table266 table252 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray113 kitchen)
     (at tray224 kitchen)
     (at tray420 kitchen)
     (at_kitchen_bread bread42)
     (at_kitchen_bread bread453)
     (at_kitchen_content content176)
     (at_kitchen_content content68)
     (not_allergic_gluten child45)
     (not_allergic_gluten child297)
     (waiting child45 table169)
     (waiting child297 table252)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child45)
     (served child297)
    )
  )
)
