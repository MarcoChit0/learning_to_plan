; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 667507

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child337 child210 - child
    bread431 bread312 - bread-portion
    content170 content174 - content-portion
    tray277 - tray
    table462 table496 table453 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray277 kitchen)
     (at_kitchen_bread bread431)
     (at_kitchen_bread bread312)
     (at_kitchen_content content170)
     (at_kitchen_content content174)
     (no_gluten_bread bread431)
     (no_gluten_content content174)
     (allergic_gluten child210)
     (not_allergic_gluten child337)
     (waiting child337 table462)
     (waiting child210 table453)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child337)
     (served child210)
    )
  )
)
