; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 190646

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child124 child69 - child
    bread172 bread233 - bread-portion
    content29 content7 - content-portion
    tray167 tray203 tray436 - tray
    table292 table489 table433 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray167 kitchen)
     (at tray203 kitchen)
     (at tray436 kitchen)
     (at_kitchen_bread bread172)
     (at_kitchen_bread bread233)
     (at_kitchen_content content29)
     (at_kitchen_content content7)
     (no_gluten_bread bread233)
     (no_gluten_content content7)
     (allergic_gluten child69)
     (not_allergic_gluten child124)
     (waiting child124 table489)
     (waiting child69 table292)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child124)
     (served child69)
    )
  )
)
