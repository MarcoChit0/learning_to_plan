; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 294922

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child490 child363 - child
    bread148 bread0 - bread-portion
    content259 content288 - content-portion
    tray337 tray31 tray453 - tray
    table411 table340 table157 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray337 kitchen)
     (at tray31 kitchen)
     (at tray453 kitchen)
     (at_kitchen_bread bread148)
     (at_kitchen_bread bread0)
     (at_kitchen_content content259)
     (at_kitchen_content content288)
     (no_gluten_bread bread148)
     (no_gluten_content content259)
     (allergic_gluten child490)
     (not_allergic_gluten child363)
     (waiting child490 table157)
     (waiting child363 table411)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child490)
     (served child363)
    )
  )
)
