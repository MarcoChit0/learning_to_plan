; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 978122

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child485 child192 - child
    bread121 bread497 - bread-portion
    content201 content400 - content-portion
    tray83 - tray
    table412 table402 table474 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray83 kitchen)
     (at_kitchen_bread bread121)
     (at_kitchen_bread bread497)
     (at_kitchen_content content201)
     (at_kitchen_content content400)
     (no_gluten_bread bread497)
     (no_gluten_content content201)
     (allergic_gluten child192)
     (not_allergic_gluten child485)
     (waiting child485 table402)
     (waiting child192 table402)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child485)
     (served child192)
    )
  )
)
