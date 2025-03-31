; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 718772

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child67 child312 - child
    bread356 bread223 - bread-portion
    content58 content494 - content-portion
    tray283 tray472 - tray
    table240 table421 table324 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray283 kitchen)
     (at tray472 kitchen)
     (at_kitchen_bread bread356)
     (at_kitchen_bread bread223)
     (at_kitchen_content content58)
     (at_kitchen_content content494)
     (not_allergic_gluten child312)
     (not_allergic_gluten child67)
     (waiting child67 table240)
     (waiting child312 table240)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child67)
     (served child312)
    )
  )
)
