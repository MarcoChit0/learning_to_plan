; child-snack task with 3 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 115011

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child92 child135 child81 - child
    bread362 bread206 bread43 - bread-portion
    content309 content144 content277 - content-portion
    tray215 tray106 tray189 - tray
    table223 table298 table449 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray215 kitchen)
     (at tray106 kitchen)
     (at tray189 kitchen)
     (at_kitchen_bread bread362)
     (at_kitchen_bread bread206)
     (at_kitchen_bread bread43)
     (at_kitchen_content content309)
     (at_kitchen_content content144)
     (at_kitchen_content content277)
     (no_gluten_bread bread362)
     (no_gluten_content content309)
     (allergic_gluten child81)
     (not_allergic_gluten child92)
     (not_allergic_gluten child135)
     (waiting child92 table223)
     (waiting child135 table298)
     (waiting child81 table223)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child92)
     (served child135)
     (served child81)
    )
  )
)
