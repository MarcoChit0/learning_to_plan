; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 648420

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child478 child118 - child
    bread351 bread480 - bread-portion
    content385 content333 - content-portion
    tray211 tray362 - tray
    table370 table178 table455 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray211 kitchen)
     (at tray362 kitchen)
     (at_kitchen_bread bread351)
     (at_kitchen_bread bread480)
     (at_kitchen_content content385)
     (at_kitchen_content content333)
     (not_allergic_gluten child478)
     (not_allergic_gluten child118)
     (waiting child478 table455)
     (waiting child118 table455)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child478)
     (served child118)
    )
  )
)
