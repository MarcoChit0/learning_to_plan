; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 767396

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child416 child92 - child
    bread144 bread28 - bread-portion
    content6 content19 - content-portion
    tray468 - tray
    table387 table323 table345 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray468 kitchen)
     (at_kitchen_bread bread144)
     (at_kitchen_bread bread28)
     (at_kitchen_content content6)
     (at_kitchen_content content19)
     (no_gluten_bread bread28)
     (no_gluten_content content6)
     (allergic_gluten child92)
     (not_allergic_gluten child416)
     (waiting child416 table387)
     (waiting child92 table387)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child416)
     (served child92)
    )
  )
)
