; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 730937

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child244 child122 - child
    bread399 bread281 - bread-portion
    content283 content194 - content-portion
    tray329 - tray
    table14 table69 table184 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray329 kitchen)
     (at_kitchen_bread bread399)
     (at_kitchen_bread bread281)
     (at_kitchen_content content283)
     (at_kitchen_content content194)
     (no_gluten_bread bread281)
     (no_gluten_content content283)
     (allergic_gluten child244)
     (not_allergic_gluten child122)
     (waiting child244 table69)
     (waiting child122 table69)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child244)
     (served child122)
    )
  )
)
