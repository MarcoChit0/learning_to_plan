; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 482334

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child228 child371 - child
    bread466 bread150 - bread-portion
    content371 content134 - content-portion
    tray309 - tray
    table147 table337 table390 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray309 kitchen)
     (at_kitchen_bread bread466)
     (at_kitchen_bread bread150)
     (at_kitchen_content content371)
     (at_kitchen_content content134)
     (no_gluten_bread bread466)
     (no_gluten_content content371)
     (allergic_gluten child228)
     (not_allergic_gluten child371)
     (waiting child228 table337)
     (waiting child371 table147)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child228)
     (served child371)
    )
  )
)
