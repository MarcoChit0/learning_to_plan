; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 254902

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child34 - child
    bread15 - bread-portion
    content426 - content-portion
    tray160 tray159 tray297 tray31 - tray
    table421 table27 table494 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray160 kitchen)
     (at tray159 kitchen)
     (at tray297 kitchen)
     (at tray31 kitchen)
     (at_kitchen_bread bread15)
     (at_kitchen_content content426)
     (not_allergic_gluten child34)
     (waiting child34 table27)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child34)
    )
  )
)
