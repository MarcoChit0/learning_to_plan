; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 390856

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child262 child234 - child
    bread439 bread122 - bread-portion
    content287 content489 - content-portion
    tray425 tray470 - tray
    table349 table223 table468 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray425 kitchen)
     (at tray470 kitchen)
     (at_kitchen_bread bread439)
     (at_kitchen_bread bread122)
     (at_kitchen_content content287)
     (at_kitchen_content content489)
     (not_allergic_gluten child234)
     (not_allergic_gluten child262)
     (waiting child262 table468)
     (waiting child234 table349)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child262)
     (served child234)
    )
  )
)
