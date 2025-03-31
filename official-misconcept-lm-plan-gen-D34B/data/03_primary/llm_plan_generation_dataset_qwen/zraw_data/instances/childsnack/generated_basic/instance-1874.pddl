; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 762247

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child344 child487 - child
    bread277 bread412 - bread-portion
    content195 content132 - content-portion
    tray80 tray139 - tray
    table237 table129 table270 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray80 kitchen)
     (at tray139 kitchen)
     (at_kitchen_bread bread277)
     (at_kitchen_bread bread412)
     (at_kitchen_content content195)
     (at_kitchen_content content132)
     (no_gluten_bread bread412)
     (no_gluten_content content195)
     (allergic_gluten child344)
     (not_allergic_gluten child487)
     (waiting child344 table270)
     (waiting child487 table129)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child344)
     (served child487)
    )
  )
)
