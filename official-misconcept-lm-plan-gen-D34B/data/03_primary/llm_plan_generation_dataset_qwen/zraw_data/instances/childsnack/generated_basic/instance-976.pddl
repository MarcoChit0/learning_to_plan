; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 738885

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child396 child38 child174 - child
    bread270 bread30 bread284 - bread-portion
    content243 content449 content175 - content-portion
    tray159 tray35 tray255 tray345 - tray
    table37 table215 table164 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray159 kitchen)
     (at tray35 kitchen)
     (at tray255 kitchen)
     (at tray345 kitchen)
     (at_kitchen_bread bread270)
     (at_kitchen_bread bread30)
     (at_kitchen_bread bread284)
     (at_kitchen_content content243)
     (at_kitchen_content content449)
     (at_kitchen_content content175)
     (not_allergic_gluten child174)
     (not_allergic_gluten child396)
     (not_allergic_gluten child38)
     (waiting child396 table37)
     (waiting child38 table37)
     (waiting child174 table37)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child396)
     (served child38)
     (served child174)
    )
  )
)
