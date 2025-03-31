; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 30666

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child436 child227 - child
    bread226 bread337 - bread-portion
    content406 content309 - content-portion
    tray473 - tray
    table335 table341 table339 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray473 kitchen)
     (at_kitchen_bread bread226)
     (at_kitchen_bread bread337)
     (at_kitchen_content content406)
     (at_kitchen_content content309)
     (no_gluten_bread bread226)
     (no_gluten_content content309)
     (allergic_gluten child227)
     (not_allergic_gluten child436)
     (waiting child436 table335)
     (waiting child227 table335)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child436)
     (served child227)
    )
  )
)
