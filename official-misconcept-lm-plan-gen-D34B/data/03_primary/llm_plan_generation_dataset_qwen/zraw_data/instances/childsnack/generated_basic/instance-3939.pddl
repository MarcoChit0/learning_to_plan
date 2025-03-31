; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 642154

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child490 child291 - child
    bread263 bread88 - bread-portion
    content79 content249 - content-portion
    tray316 - tray
    table356 table117 table35 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray316 kitchen)
     (at_kitchen_bread bread263)
     (at_kitchen_bread bread88)
     (at_kitchen_content content79)
     (at_kitchen_content content249)
     (no_gluten_bread bread88)
     (no_gluten_content content79)
     (allergic_gluten child291)
     (not_allergic_gluten child490)
     (waiting child490 table117)
     (waiting child291 table35)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child490)
     (served child291)
    )
  )
)
