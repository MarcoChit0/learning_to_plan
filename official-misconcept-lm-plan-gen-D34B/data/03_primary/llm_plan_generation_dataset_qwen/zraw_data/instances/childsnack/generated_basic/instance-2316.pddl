; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 994372

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child320 - child
    bread288 - bread-portion
    content466 - content-portion
    tray417 - tray
    table258 table107 table330 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray417 kitchen)
     (at_kitchen_bread bread288)
     (at_kitchen_content content466)
     (not_allergic_gluten child320)
     (waiting child320 table330)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child320)
    )
  )
)
