; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 844716

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child111 child370 - child
    bread438 bread16 - bread-portion
    content131 content450 - content-portion
    tray86 tray350 tray414 - tray
    table201 table129 table346 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray86 kitchen)
     (at tray350 kitchen)
     (at tray414 kitchen)
     (at_kitchen_bread bread438)
     (at_kitchen_bread bread16)
     (at_kitchen_content content131)
     (at_kitchen_content content450)
     (no_gluten_bread bread438)
     (no_gluten_content content450)
     (allergic_gluten child370)
     (not_allergic_gluten child111)
     (waiting child111 table346)
     (waiting child370 table346)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child111)
     (served child370)
    )
  )
)
