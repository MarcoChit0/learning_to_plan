; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 634559

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child291 child331 - child
    bread475 bread424 - bread-portion
    content160 content459 - content-portion
    tray490 - tray
    table103 table282 table319 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray490 kitchen)
     (at_kitchen_bread bread475)
     (at_kitchen_bread bread424)
     (at_kitchen_content content160)
     (at_kitchen_content content459)
     (no_gluten_bread bread424)
     (no_gluten_content content459)
     (allergic_gluten child331)
     (not_allergic_gluten child291)
     (waiting child291 table282)
     (waiting child331 table103)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child291)
     (served child331)
    )
  )
)
