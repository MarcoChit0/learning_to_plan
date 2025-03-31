; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 572670

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child366 child138 - child
    bread377 bread469 - bread-portion
    content418 content380 - content-portion
    tray212 tray360 - tray
    table292 table383 table433 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray212 kitchen)
     (at tray360 kitchen)
     (at_kitchen_bread bread377)
     (at_kitchen_bread bread469)
     (at_kitchen_content content418)
     (at_kitchen_content content380)
     (no_gluten_bread bread469)
     (no_gluten_content content418)
     (allergic_gluten child138)
     (not_allergic_gluten child366)
     (waiting child366 table383)
     (waiting child138 table383)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child366)
     (served child138)
    )
  )
)
