; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 19176

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child328 child165 - child
    bread97 bread90 - bread-portion
    content335 content370 - content-portion
    tray319 - tray
    table282 table186 table327 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray319 kitchen)
     (at_kitchen_bread bread97)
     (at_kitchen_bread bread90)
     (at_kitchen_content content335)
     (at_kitchen_content content370)
     (no_gluten_bread bread97)
     (no_gluten_content content370)
     (allergic_gluten child328)
     (not_allergic_gluten child165)
     (waiting child328 table282)
     (waiting child165 table186)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child328)
     (served child165)
    )
  )
)
