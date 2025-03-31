; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 573480

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child323 child5 - child
    bread390 bread133 - bread-portion
    content456 content105 - content-portion
    tray211 - tray
    table447 table137 table149 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray211 kitchen)
     (at_kitchen_bread bread390)
     (at_kitchen_bread bread133)
     (at_kitchen_content content456)
     (at_kitchen_content content105)
     (no_gluten_bread bread133)
     (no_gluten_content content456)
     (allergic_gluten child323)
     (not_allergic_gluten child5)
     (waiting child323 table149)
     (waiting child5 table447)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child323)
     (served child5)
    )
  )
)
