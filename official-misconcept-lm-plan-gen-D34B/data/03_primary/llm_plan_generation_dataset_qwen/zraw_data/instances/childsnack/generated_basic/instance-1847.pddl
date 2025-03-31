; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 269663

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child355 child174 - child
    bread426 bread363 - bread-portion
    content21 content421 - content-portion
    tray165 - tray
    table22 table11 table345 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray165 kitchen)
     (at_kitchen_bread bread426)
     (at_kitchen_bread bread363)
     (at_kitchen_content content21)
     (at_kitchen_content content421)
     (no_gluten_bread bread426)
     (no_gluten_content content421)
     (allergic_gluten child355)
     (not_allergic_gluten child174)
     (waiting child355 table345)
     (waiting child174 table22)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child355)
     (served child174)
    )
  )
)
