; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 815072

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child227 child175 - child
    bread72 bread445 - bread-portion
    content486 content342 - content-portion
    tray47 - tray
    table7 table355 table106 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray47 kitchen)
     (at_kitchen_bread bread72)
     (at_kitchen_bread bread445)
     (at_kitchen_content content486)
     (at_kitchen_content content342)
     (no_gluten_bread bread72)
     (no_gluten_content content342)
     (allergic_gluten child227)
     (not_allergic_gluten child175)
     (waiting child227 table106)
     (waiting child175 table355)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child227)
     (served child175)
    )
  )
)
