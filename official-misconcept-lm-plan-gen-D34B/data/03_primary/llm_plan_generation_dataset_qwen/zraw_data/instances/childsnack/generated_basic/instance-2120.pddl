; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 392107

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child490 child204 - child
    bread69 bread304 - bread-portion
    content123 content290 - content-portion
    tray33 tray452 tray63 - tray
    table413 table380 table453 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray33 kitchen)
     (at tray452 kitchen)
     (at tray63 kitchen)
     (at_kitchen_bread bread69)
     (at_kitchen_bread bread304)
     (at_kitchen_content content123)
     (at_kitchen_content content290)
     (no_gluten_bread bread69)
     (no_gluten_content content290)
     (allergic_gluten child490)
     (not_allergic_gluten child204)
     (waiting child490 table413)
     (waiting child204 table380)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child490)
     (served child204)
    )
  )
)
