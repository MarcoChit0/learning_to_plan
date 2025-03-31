; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 517451

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child338 child296 - child
    bread132 bread462 - bread-portion
    content428 content215 - content-portion
    tray169 tray67 - tray
    table199 table234 table57 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray169 kitchen)
     (at tray67 kitchen)
     (at_kitchen_bread bread132)
     (at_kitchen_bread bread462)
     (at_kitchen_content content428)
     (at_kitchen_content content215)
     (no_gluten_bread bread462)
     (no_gluten_content content428)
     (allergic_gluten child338)
     (not_allergic_gluten child296)
     (waiting child338 table57)
     (waiting child296 table199)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child338)
     (served child296)
    )
  )
)
