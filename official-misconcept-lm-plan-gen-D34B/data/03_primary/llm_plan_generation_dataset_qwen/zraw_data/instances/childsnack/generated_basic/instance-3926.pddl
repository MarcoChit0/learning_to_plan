; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 548858

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child280 child208 - child
    bread177 bread249 - bread-portion
    content349 content442 - content-portion
    tray62 - tray
    table16 table76 table340 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray62 kitchen)
     (at_kitchen_bread bread177)
     (at_kitchen_bread bread249)
     (at_kitchen_content content349)
     (at_kitchen_content content442)
     (no_gluten_bread bread249)
     (no_gluten_content content349)
     (allergic_gluten child208)
     (not_allergic_gluten child280)
     (waiting child280 table340)
     (waiting child208 table76)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child280)
     (served child208)
    )
  )
)
