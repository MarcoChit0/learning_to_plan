; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 299803

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child228 child159 - child
    bread279 bread275 - bread-portion
    content10 content195 - content-portion
    tray448 - tray
    table27 table118 table283 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray448 kitchen)
     (at_kitchen_bread bread279)
     (at_kitchen_bread bread275)
     (at_kitchen_content content10)
     (at_kitchen_content content195)
     (no_gluten_bread bread279)
     (no_gluten_content content10)
     (allergic_gluten child159)
     (not_allergic_gluten child228)
     (waiting child228 table283)
     (waiting child159 table118)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child228)
     (served child159)
    )
  )
)
