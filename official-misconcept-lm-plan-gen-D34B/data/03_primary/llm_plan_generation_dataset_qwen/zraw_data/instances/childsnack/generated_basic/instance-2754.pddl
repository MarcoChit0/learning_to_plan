; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 270799

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child66 child209 - child
    bread39 bread392 - bread-portion
    content23 content200 - content-portion
    tray393 - tray
    table403 table68 table45 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray393 kitchen)
     (at_kitchen_bread bread39)
     (at_kitchen_bread bread392)
     (at_kitchen_content content23)
     (at_kitchen_content content200)
     (no_gluten_bread bread39)
     (no_gluten_content content23)
     (allergic_gluten child209)
     (not_allergic_gluten child66)
     (waiting child66 table45)
     (waiting child209 table68)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child66)
     (served child209)
    )
  )
)
