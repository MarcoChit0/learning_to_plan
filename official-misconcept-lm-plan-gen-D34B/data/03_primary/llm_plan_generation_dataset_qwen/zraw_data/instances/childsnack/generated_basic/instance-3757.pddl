; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 852851

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child127 - child
    bread406 - bread-portion
    content80 - content-portion
    tray492 tray281 tray93 - tray
    table87 table60 table365 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray492 kitchen)
     (at tray281 kitchen)
     (at tray93 kitchen)
     (at_kitchen_bread bread406)
     (at_kitchen_content content80)
     (not_allergic_gluten child127)
     (waiting child127 table60)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child127)
    )
  )
)
