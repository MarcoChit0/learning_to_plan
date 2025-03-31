; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 392776

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child352 child445 - child
    bread177 bread226 - bread-portion
    content239 content398 - content-portion
    tray429 - tray
    table457 table198 table448 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray429 kitchen)
     (at_kitchen_bread bread177)
     (at_kitchen_bread bread226)
     (at_kitchen_content content239)
     (at_kitchen_content content398)
     (no_gluten_bread bread226)
     (no_gluten_content content239)
     (allergic_gluten child352)
     (not_allergic_gluten child445)
     (waiting child352 table198)
     (waiting child445 table448)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child352)
     (served child445)
    )
  )
)
