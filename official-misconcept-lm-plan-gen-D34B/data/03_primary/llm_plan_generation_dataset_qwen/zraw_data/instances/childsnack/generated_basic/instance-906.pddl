; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 542890

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child362 child225 child91 - child
    bread343 bread358 bread276 - bread-portion
    content85 content243 content411 - content-portion
    tray258 - tray
    table452 table192 table262 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray258 kitchen)
     (at_kitchen_bread bread343)
     (at_kitchen_bread bread358)
     (at_kitchen_bread bread276)
     (at_kitchen_content content85)
     (at_kitchen_content content243)
     (at_kitchen_content content411)
     (not_allergic_gluten child362)
     (not_allergic_gluten child91)
     (not_allergic_gluten child225)
     (waiting child362 table192)
     (waiting child225 table262)
     (waiting child91 table192)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child362)
     (served child225)
     (served child91)
    )
  )
)
