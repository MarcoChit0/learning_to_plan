; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 356381

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child124 child109 - child
    bread126 bread496 - bread-portion
    content466 content350 - content-portion
    tray266 tray1 tray292 tray486 - tray
    table329 table166 table163 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray266 kitchen)
     (at tray1 kitchen)
     (at tray292 kitchen)
     (at tray486 kitchen)
     (at_kitchen_bread bread126)
     (at_kitchen_bread bread496)
     (at_kitchen_content content466)
     (at_kitchen_content content350)
     (no_gluten_bread bread126)
     (no_gluten_content content466)
     (allergic_gluten child109)
     (not_allergic_gluten child124)
     (waiting child124 table329)
     (waiting child109 table166)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child124)
     (served child109)
    )
  )
)
