; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 372355

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child389 child445 - child
    bread295 bread285 - bread-portion
    content325 content194 - content-portion
    tray79 tray177 - tray
    table188 table475 table250 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray79 kitchen)
     (at tray177 kitchen)
     (at_kitchen_bread bread295)
     (at_kitchen_bread bread285)
     (at_kitchen_content content325)
     (at_kitchen_content content194)
     (no_gluten_bread bread285)
     (no_gluten_content content325)
     (allergic_gluten child389)
     (not_allergic_gluten child445)
     (waiting child389 table475)
     (waiting child445 table188)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child389)
     (served child445)
    )
  )
)
