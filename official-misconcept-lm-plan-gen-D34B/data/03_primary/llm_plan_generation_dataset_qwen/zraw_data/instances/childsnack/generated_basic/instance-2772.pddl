; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 963828

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child445 child277 - child
    bread409 bread262 - bread-portion
    content412 content247 - content-portion
    tray408 - tray
    table449 table8 table38 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray408 kitchen)
     (at_kitchen_bread bread409)
     (at_kitchen_bread bread262)
     (at_kitchen_content content412)
     (at_kitchen_content content247)
     (no_gluten_bread bread409)
     (no_gluten_content content247)
     (allergic_gluten child277)
     (not_allergic_gluten child445)
     (waiting child445 table449)
     (waiting child277 table8)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child445)
     (served child277)
    )
  )
)
