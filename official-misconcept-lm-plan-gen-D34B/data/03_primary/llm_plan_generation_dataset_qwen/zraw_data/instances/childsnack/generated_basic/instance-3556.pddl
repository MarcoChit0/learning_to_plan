; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 184119

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child367 child260 - child
    bread453 bread312 - bread-portion
    content120 content29 - content-portion
    tray138 tray128 - tray
    table97 table380 table114 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray138 kitchen)
     (at tray128 kitchen)
     (at_kitchen_bread bread453)
     (at_kitchen_bread bread312)
     (at_kitchen_content content120)
     (at_kitchen_content content29)
     (not_allergic_gluten child260)
     (not_allergic_gluten child367)
     (waiting child367 table380)
     (waiting child260 table97)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child367)
     (served child260)
    )
  )
)
