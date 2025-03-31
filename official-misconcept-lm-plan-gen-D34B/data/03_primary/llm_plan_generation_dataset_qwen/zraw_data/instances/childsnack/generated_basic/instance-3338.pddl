; child-snack task with 2 children and 0.71 gluten factor 
; constant factor of 1.7
; random seed: 99739

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child109 child208 - child
    bread389 bread354 - bread-portion
    content209 content279 - content-portion
    tray0 tray6 tray370 tray285 - tray
    table439 table497 table451 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray0 kitchen)
     (at tray6 kitchen)
     (at tray370 kitchen)
     (at tray285 kitchen)
     (at_kitchen_bread bread389)
     (at_kitchen_bread bread354)
     (at_kitchen_content content209)
     (at_kitchen_content content279)
     (no_gluten_bread bread389)
     (no_gluten_content content209)
     (allergic_gluten child109)
     (not_allergic_gluten child208)
     (waiting child109 table439)
     (waiting child208 table451)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child109)
     (served child208)
    )
  )
)
