; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 535498

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child256 child404 - child
    bread272 bread27 - bread-portion
    content445 content176 - content-portion
    tray23 tray178 tray90 - tray
    table68 table168 table416 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray23 kitchen)
     (at tray178 kitchen)
     (at tray90 kitchen)
     (at_kitchen_bread bread272)
     (at_kitchen_bread bread27)
     (at_kitchen_content content445)
     (at_kitchen_content content176)
     (no_gluten_bread bread272)
     (no_gluten_content content176)
     (allergic_gluten child404)
     (not_allergic_gluten child256)
     (waiting child256 table68)
     (waiting child404 table168)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child256)
     (served child404)
    )
  )
)
