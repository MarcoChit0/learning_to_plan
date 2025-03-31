; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 382217

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child324 child235 - child
    bread118 bread107 - bread-portion
    content152 content269 - content-portion
    tray350 - tray
    table478 table186 table11 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray350 kitchen)
     (at_kitchen_bread bread118)
     (at_kitchen_bread bread107)
     (at_kitchen_content content152)
     (at_kitchen_content content269)
     (no_gluten_bread bread118)
     (no_gluten_content content269)
     (allergic_gluten child324)
     (not_allergic_gluten child235)
     (waiting child324 table478)
     (waiting child235 table478)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child324)
     (served child235)
    )
  )
)
