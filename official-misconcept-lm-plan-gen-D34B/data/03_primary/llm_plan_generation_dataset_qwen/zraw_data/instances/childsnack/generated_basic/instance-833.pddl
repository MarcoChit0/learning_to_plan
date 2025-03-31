; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 56196

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child350 child84 - child
    bread482 bread140 - bread-portion
    content227 content288 - content-portion
    tray436 tray92 - tray
    table174 table81 table460 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray436 kitchen)
     (at tray92 kitchen)
     (at_kitchen_bread bread482)
     (at_kitchen_bread bread140)
     (at_kitchen_content content227)
     (at_kitchen_content content288)
     (no_gluten_bread bread482)
     (no_gluten_content content288)
     (allergic_gluten child84)
     (not_allergic_gluten child350)
     (waiting child350 table460)
     (waiting child84 table174)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child350)
     (served child84)
    )
  )
)
