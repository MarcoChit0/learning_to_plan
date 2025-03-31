; child-snack task with 2 children and 0.61 gluten factor 
; constant factor of 1.3
; random seed: 487411

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child266 child329 - child
    bread428 bread487 - bread-portion
    content397 content481 - content-portion
    tray222 tray411 tray78 tray202 - tray
    table414 table390 table463 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray222 kitchen)
     (at tray411 kitchen)
     (at tray78 kitchen)
     (at tray202 kitchen)
     (at_kitchen_bread bread428)
     (at_kitchen_bread bread487)
     (at_kitchen_content content397)
     (at_kitchen_content content481)
     (no_gluten_bread bread428)
     (no_gluten_content content397)
     (allergic_gluten child266)
     (not_allergic_gluten child329)
     (waiting child266 table463)
     (waiting child329 table414)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child266)
     (served child329)
    )
  )
)
