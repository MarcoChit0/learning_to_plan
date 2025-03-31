; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 714482

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child227 child122 child167 - child
    bread49 bread479 bread361 - bread-portion
    content189 content298 content321 - content-portion
    tray355 - tray
    table171 table270 table398 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray355 kitchen)
     (at_kitchen_bread bread49)
     (at_kitchen_bread bread479)
     (at_kitchen_bread bread361)
     (at_kitchen_content content189)
     (at_kitchen_content content298)
     (at_kitchen_content content321)
     (not_allergic_gluten child122)
     (not_allergic_gluten child227)
     (not_allergic_gluten child167)
     (waiting child227 table398)
     (waiting child122 table171)
     (waiting child167 table398)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child227)
     (served child122)
     (served child167)
    )
  )
)
