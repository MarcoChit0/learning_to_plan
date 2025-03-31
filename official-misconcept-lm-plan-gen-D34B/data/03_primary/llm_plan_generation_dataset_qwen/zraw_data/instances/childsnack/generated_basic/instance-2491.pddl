; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 910541

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child7 child40 - child
    bread180 bread121 - bread-portion
    content29 content189 - content-portion
    tray414 tray68 - tray
    table54 table300 table497 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray414 kitchen)
     (at tray68 kitchen)
     (at_kitchen_bread bread180)
     (at_kitchen_bread bread121)
     (at_kitchen_content content29)
     (at_kitchen_content content189)
     (not_allergic_gluten child40)
     (not_allergic_gluten child7)
     (waiting child7 table54)
     (waiting child40 table54)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child7)
     (served child40)
    )
  )
)
