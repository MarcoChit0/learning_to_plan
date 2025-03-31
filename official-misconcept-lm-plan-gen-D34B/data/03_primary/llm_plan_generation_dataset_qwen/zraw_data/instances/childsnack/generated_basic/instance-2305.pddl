; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 766069

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child393 child459 - child
    bread124 bread478 - bread-portion
    content481 content449 - content-portion
    tray71 - tray
    table409 table473 table141 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray71 kitchen)
     (at_kitchen_bread bread124)
     (at_kitchen_bread bread478)
     (at_kitchen_content content481)
     (at_kitchen_content content449)
     (no_gluten_bread bread478)
     (no_gluten_content content449)
     (allergic_gluten child459)
     (not_allergic_gluten child393)
     (waiting child393 table473)
     (waiting child459 table409)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child393)
     (served child459)
    )
  )
)
