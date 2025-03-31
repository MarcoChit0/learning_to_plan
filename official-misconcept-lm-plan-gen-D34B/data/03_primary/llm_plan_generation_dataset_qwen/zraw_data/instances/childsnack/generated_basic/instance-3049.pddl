; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 111998

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child128 child124 - child
    bread112 bread433 - bread-portion
    content41 content27 - content-portion
    tray152 tray353 tray36 - tray
    table254 table193 table478 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray152 kitchen)
     (at tray353 kitchen)
     (at tray36 kitchen)
     (at_kitchen_bread bread112)
     (at_kitchen_bread bread433)
     (at_kitchen_content content41)
     (at_kitchen_content content27)
     (no_gluten_bread bread112)
     (no_gluten_content content41)
     (allergic_gluten child124)
     (not_allergic_gluten child128)
     (waiting child128 table478)
     (waiting child124 table193)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child128)
     (served child124)
    )
  )
)
