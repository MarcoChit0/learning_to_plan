; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 728282

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child421 child407 - child
    bread16 bread378 - bread-portion
    content129 content410 - content-portion
    tray306 - tray
    table80 table382 table127 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray306 kitchen)
     (at_kitchen_bread bread16)
     (at_kitchen_bread bread378)
     (at_kitchen_content content129)
     (at_kitchen_content content410)
     (no_gluten_bread bread378)
     (no_gluten_content content410)
     (allergic_gluten child421)
     (not_allergic_gluten child407)
     (waiting child421 table382)
     (waiting child407 table127)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child421)
     (served child407)
    )
  )
)
