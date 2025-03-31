; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 28394

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child227 child455 - child
    bread234 bread266 - bread-portion
    content138 content314 - content-portion
    tray362 tray268 tray113 - tray
    table369 table169 table417 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray362 kitchen)
     (at tray268 kitchen)
     (at tray113 kitchen)
     (at_kitchen_bread bread234)
     (at_kitchen_bread bread266)
     (at_kitchen_content content138)
     (at_kitchen_content content314)
     (not_allergic_gluten child455)
     (not_allergic_gluten child227)
     (waiting child227 table169)
     (waiting child455 table369)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child227)
     (served child455)
    )
  )
)
