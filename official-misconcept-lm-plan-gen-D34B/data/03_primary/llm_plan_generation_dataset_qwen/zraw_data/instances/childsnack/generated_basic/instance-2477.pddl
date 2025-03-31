; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 602946

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child266 child431 - child
    bread19 bread453 - bread-portion
    content169 content253 - content-portion
    tray375 tray490 - tray
    table59 table236 table309 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray375 kitchen)
     (at tray490 kitchen)
     (at_kitchen_bread bread19)
     (at_kitchen_bread bread453)
     (at_kitchen_content content169)
     (at_kitchen_content content253)
     (not_allergic_gluten child266)
     (not_allergic_gluten child431)
     (waiting child266 table236)
     (waiting child431 table309)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child266)
     (served child431)
    )
  )
)
