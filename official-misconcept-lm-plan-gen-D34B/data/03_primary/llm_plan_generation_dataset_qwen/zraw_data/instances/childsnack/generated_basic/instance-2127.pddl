; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 191124

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child478 child437 - child
    bread216 bread315 - bread-portion
    content159 content97 - content-portion
    tray230 tray103 tray290 - tray
    table14 table453 table446 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray230 kitchen)
     (at tray103 kitchen)
     (at tray290 kitchen)
     (at_kitchen_bread bread216)
     (at_kitchen_bread bread315)
     (at_kitchen_content content159)
     (at_kitchen_content content97)
     (no_gluten_bread bread315)
     (no_gluten_content content159)
     (allergic_gluten child478)
     (not_allergic_gluten child437)
     (waiting child478 table14)
     (waiting child437 table446)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child478)
     (served child437)
    )
  )
)
