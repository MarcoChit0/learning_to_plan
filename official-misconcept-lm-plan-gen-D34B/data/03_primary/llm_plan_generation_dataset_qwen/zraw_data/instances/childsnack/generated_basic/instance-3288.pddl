; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 554232

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child110 child366 - child
    bread40 bread420 - bread-portion
    content203 content402 - content-portion
    tray405 tray349 tray88 tray264 - tray
    table105 table229 table186 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray405 kitchen)
     (at tray349 kitchen)
     (at tray88 kitchen)
     (at tray264 kitchen)
     (at_kitchen_bread bread40)
     (at_kitchen_bread bread420)
     (at_kitchen_content content203)
     (at_kitchen_content content402)
     (no_gluten_bread bread420)
     (no_gluten_content content402)
     (allergic_gluten child110)
     (not_allergic_gluten child366)
     (waiting child110 table186)
     (waiting child366 table105)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child110)
     (served child366)
    )
  )
)
