; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 106703

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child295 - child
    bread450 - bread-portion
    content341 - content-portion
    tray86 tray402 tray186 - tray
    table168 table210 table105 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray86 kitchen)
     (at tray402 kitchen)
     (at tray186 kitchen)
     (at_kitchen_bread bread450)
     (at_kitchen_content content341)
     (not_allergic_gluten child295)
     (waiting child295 table210)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child295)
    )
  )
)
