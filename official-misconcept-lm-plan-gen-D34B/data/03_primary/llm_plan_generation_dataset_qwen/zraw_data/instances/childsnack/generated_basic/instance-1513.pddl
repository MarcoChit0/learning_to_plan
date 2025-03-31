; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 61420

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child409 child295 - child
    bread222 bread226 - bread-portion
    content445 content158 - content-portion
    tray422 tray2 tray278 - tray
    table456 table205 table219 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray422 kitchen)
     (at tray2 kitchen)
     (at tray278 kitchen)
     (at_kitchen_bread bread222)
     (at_kitchen_bread bread226)
     (at_kitchen_content content445)
     (at_kitchen_content content158)
     (no_gluten_bread bread222)
     (no_gluten_content content158)
     (allergic_gluten child409)
     (not_allergic_gluten child295)
     (waiting child409 table219)
     (waiting child295 table205)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child409)
     (served child295)
    )
  )
)
