; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 267551

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child117 child317 child394 - child
    bread129 bread188 bread127 - bread-portion
    content33 content482 content182 - content-portion
    tray349 - tray
    table459 table14 table451 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray349 kitchen)
     (at_kitchen_bread bread129)
     (at_kitchen_bread bread188)
     (at_kitchen_bread bread127)
     (at_kitchen_content content33)
     (at_kitchen_content content482)
     (at_kitchen_content content182)
     (not_allergic_gluten child117)
     (not_allergic_gluten child394)
     (not_allergic_gluten child317)
     (waiting child117 table459)
     (waiting child317 table451)
     (waiting child394 table459)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child117)
     (served child317)
     (served child394)
    )
  )
)
