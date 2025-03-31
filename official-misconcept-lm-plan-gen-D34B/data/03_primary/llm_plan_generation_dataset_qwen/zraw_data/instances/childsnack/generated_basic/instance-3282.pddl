; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 344025

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child237 child45 - child
    bread433 bread140 - bread-portion
    content377 content324 - content-portion
    tray485 tray13 tray458 tray155 - tray
    table342 table61 table103 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray485 kitchen)
     (at tray13 kitchen)
     (at tray458 kitchen)
     (at tray155 kitchen)
     (at_kitchen_bread bread433)
     (at_kitchen_bread bread140)
     (at_kitchen_content content377)
     (at_kitchen_content content324)
     (no_gluten_bread bread433)
     (no_gluten_content content377)
     (allergic_gluten child237)
     (not_allergic_gluten child45)
     (waiting child237 table103)
     (waiting child45 table61)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child237)
     (served child45)
    )
  )
)
