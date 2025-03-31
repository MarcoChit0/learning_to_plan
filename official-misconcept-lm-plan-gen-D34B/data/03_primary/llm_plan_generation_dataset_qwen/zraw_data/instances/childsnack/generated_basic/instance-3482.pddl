; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 820582

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child312 child126 - child
    bread295 bread281 - bread-portion
    content0 content13 - content-portion
    tray104 tray114 - tray
    table291 table89 table15 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray104 kitchen)
     (at tray114 kitchen)
     (at_kitchen_bread bread295)
     (at_kitchen_bread bread281)
     (at_kitchen_content content0)
     (at_kitchen_content content13)
     (no_gluten_bread bread281)
     (no_gluten_content content0)
     (allergic_gluten child126)
     (not_allergic_gluten child312)
     (waiting child312 table291)
     (waiting child126 table291)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child312)
     (served child126)
    )
  )
)
