; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 231256

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child177 child179 - child
    bread367 bread440 - bread-portion
    content203 content284 - content-portion
    tray273 tray196 tray122 - tray
    table253 table456 table412 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray273 kitchen)
     (at tray196 kitchen)
     (at tray122 kitchen)
     (at_kitchen_bread bread367)
     (at_kitchen_bread bread440)
     (at_kitchen_content content203)
     (at_kitchen_content content284)
     (no_gluten_bread bread440)
     (no_gluten_content content203)
     (allergic_gluten child177)
     (not_allergic_gluten child179)
     (waiting child177 table253)
     (waiting child179 table456)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child177)
     (served child179)
    )
  )
)
