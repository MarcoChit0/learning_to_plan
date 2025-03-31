; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 56211

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child362 child132 - child
    bread23 bread27 - bread-portion
    content48 content424 - content-portion
    tray306 - tray
    table174 table189 table110 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray306 kitchen)
     (at_kitchen_bread bread23)
     (at_kitchen_bread bread27)
     (at_kitchen_content content48)
     (at_kitchen_content content424)
     (no_gluten_bread bread27)
     (no_gluten_content content424)
     (allergic_gluten child132)
     (not_allergic_gluten child362)
     (waiting child362 table110)
     (waiting child132 table110)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child362)
     (served child132)
    )
  )
)
