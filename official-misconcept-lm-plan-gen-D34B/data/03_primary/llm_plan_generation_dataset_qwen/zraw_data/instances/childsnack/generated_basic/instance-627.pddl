; child-snack task with 3 children and 0.7 gluten factor 
; constant factor of 1.7
; random seed: 696957

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child11 child261 child382 - child
    bread479 bread346 bread447 - bread-portion
    content0 content223 content321 - content-portion
    tray120 tray211 - tray
    table408 table129 table340 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray120 kitchen)
     (at tray211 kitchen)
     (at_kitchen_bread bread479)
     (at_kitchen_bread bread346)
     (at_kitchen_bread bread447)
     (at_kitchen_content content0)
     (at_kitchen_content content223)
     (at_kitchen_content content321)
     (no_gluten_bread bread447)
     (no_gluten_bread bread346)
     (no_gluten_content content0)
     (no_gluten_content content321)
     (allergic_gluten child261)
     (allergic_gluten child11)
     (not_allergic_gluten child382)
     (waiting child11 table129)
     (waiting child261 table129)
     (waiting child382 table340)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child11)
     (served child261)
     (served child382)
    )
  )
)
