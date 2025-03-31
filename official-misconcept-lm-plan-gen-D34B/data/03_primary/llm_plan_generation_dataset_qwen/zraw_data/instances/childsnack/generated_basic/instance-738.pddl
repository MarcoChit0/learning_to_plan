; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 192727

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child352 child138 - child
    bread455 bread384 - bread-portion
    content13 content430 - content-portion
    tray307 tray33 - tray
    table209 table455 table166 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray307 kitchen)
     (at tray33 kitchen)
     (at_kitchen_bread bread455)
     (at_kitchen_bread bread384)
     (at_kitchen_content content13)
     (at_kitchen_content content430)
     (no_gluten_bread bread384)
     (no_gluten_content content13)
     (allergic_gluten child138)
     (not_allergic_gluten child352)
     (waiting child352 table455)
     (waiting child138 table166)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child352)
     (served child138)
    )
  )
)
