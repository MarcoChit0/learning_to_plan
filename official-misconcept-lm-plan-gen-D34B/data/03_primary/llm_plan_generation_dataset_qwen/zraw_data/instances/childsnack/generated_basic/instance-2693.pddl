; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 149722

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child451 child194 child2 - child
    bread326 bread340 bread381 - bread-portion
    content294 content267 content304 - content-portion
    tray189 tray369 tray207 - tray
    table302 table145 table240 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray189 kitchen)
     (at tray369 kitchen)
     (at tray207 kitchen)
     (at_kitchen_bread bread326)
     (at_kitchen_bread bread340)
     (at_kitchen_bread bread381)
     (at_kitchen_content content294)
     (at_kitchen_content content267)
     (at_kitchen_content content304)
     (not_allergic_gluten child2)
     (not_allergic_gluten child194)
     (not_allergic_gluten child451)
     (waiting child451 table145)
     (waiting child194 table302)
     (waiting child2 table240)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child451)
     (served child194)
     (served child2)
    )
  )
)
