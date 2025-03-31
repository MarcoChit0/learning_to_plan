; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 107284

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child102 child375 - child
    bread252 bread398 - bread-portion
    content70 content2 - content-portion
    tray90 - tray
    table71 table209 table327 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray90 kitchen)
     (at_kitchen_bread bread252)
     (at_kitchen_bread bread398)
     (at_kitchen_content content70)
     (at_kitchen_content content2)
     (no_gluten_bread bread252)
     (no_gluten_content content2)
     (allergic_gluten child102)
     (not_allergic_gluten child375)
     (waiting child102 table327)
     (waiting child375 table71)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child102)
     (served child375)
    )
  )
)
