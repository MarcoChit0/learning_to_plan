; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 316561

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child472 child3 - child
    bread247 bread426 - bread-portion
    content341 content471 - content-portion
    tray455 tray74 - tray
    table76 table304 table244 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray455 kitchen)
     (at tray74 kitchen)
     (at_kitchen_bread bread247)
     (at_kitchen_bread bread426)
     (at_kitchen_content content341)
     (at_kitchen_content content471)
     (no_gluten_bread bread247)
     (no_gluten_content content341)
     (allergic_gluten child472)
     (not_allergic_gluten child3)
     (waiting child472 table244)
     (waiting child3 table76)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child472)
     (served child3)
    )
  )
)
