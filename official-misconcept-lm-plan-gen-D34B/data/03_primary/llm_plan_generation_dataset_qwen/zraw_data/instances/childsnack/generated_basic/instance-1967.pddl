; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 893479

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child79 child55 - child
    bread456 bread210 - bread-portion
    content355 content140 - content-portion
    tray115 - tray
    table80 table168 table321 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray115 kitchen)
     (at_kitchen_bread bread456)
     (at_kitchen_bread bread210)
     (at_kitchen_content content355)
     (at_kitchen_content content140)
     (no_gluten_bread bread456)
     (no_gluten_content content355)
     (allergic_gluten child55)
     (not_allergic_gluten child79)
     (waiting child79 table80)
     (waiting child55 table168)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child79)
     (served child55)
    )
  )
)
