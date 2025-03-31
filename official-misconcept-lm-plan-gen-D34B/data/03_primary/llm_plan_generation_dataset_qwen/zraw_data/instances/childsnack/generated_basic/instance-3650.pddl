; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 36057

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child57 child200 - child
    bread415 bread195 - bread-portion
    content367 content216 - content-portion
    tray489 tray117 tray108 tray336 - tray
    table341 table469 table207 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray489 kitchen)
     (at tray117 kitchen)
     (at tray108 kitchen)
     (at tray336 kitchen)
     (at_kitchen_bread bread415)
     (at_kitchen_bread bread195)
     (at_kitchen_content content367)
     (at_kitchen_content content216)
     (no_gluten_bread bread415)
     (no_gluten_content content216)
     (allergic_gluten child200)
     (not_allergic_gluten child57)
     (waiting child57 table341)
     (waiting child200 table469)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child57)
     (served child200)
    )
  )
)
