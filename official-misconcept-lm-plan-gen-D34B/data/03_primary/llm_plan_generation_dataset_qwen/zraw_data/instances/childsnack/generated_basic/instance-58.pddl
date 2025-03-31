; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 203200

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child320 child309 - child
    bread334 bread469 - bread-portion
    content131 content93 - content-portion
    tray392 tray76 - tray
    table394 table466 table133 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray392 kitchen)
     (at tray76 kitchen)
     (at_kitchen_bread bread334)
     (at_kitchen_bread bread469)
     (at_kitchen_content content131)
     (at_kitchen_content content93)
     (not_allergic_gluten child309)
     (not_allergic_gluten child320)
     (waiting child320 table133)
     (waiting child309 table394)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child320)
     (served child309)
    )
  )
)
