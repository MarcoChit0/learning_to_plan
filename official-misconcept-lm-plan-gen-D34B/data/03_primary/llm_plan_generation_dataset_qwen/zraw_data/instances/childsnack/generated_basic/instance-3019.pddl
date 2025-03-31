; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 735595

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child362 child31 - child
    bread147 bread472 - bread-portion
    content217 content189 - content-portion
    tray160 tray92 tray391 tray112 - tray
    table480 table90 table318 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray160 kitchen)
     (at tray92 kitchen)
     (at tray391 kitchen)
     (at tray112 kitchen)
     (at_kitchen_bread bread147)
     (at_kitchen_bread bread472)
     (at_kitchen_content content217)
     (at_kitchen_content content189)
     (not_allergic_gluten child31)
     (not_allergic_gluten child362)
     (waiting child362 table90)
     (waiting child31 table90)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child362)
     (served child31)
    )
  )
)
