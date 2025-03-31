; child-snack task with 3 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 503030

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child98 child469 child57 - child
    bread477 bread160 bread310 - bread-portion
    content449 content343 content19 - content-portion
    tray219 tray126 tray163 - tray
    table99 table4 table409 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray219 kitchen)
     (at tray126 kitchen)
     (at tray163 kitchen)
     (at_kitchen_bread bread477)
     (at_kitchen_bread bread160)
     (at_kitchen_bread bread310)
     (at_kitchen_content content449)
     (at_kitchen_content content343)
     (at_kitchen_content content19)
     (no_gluten_bread bread477)
     (no_gluten_content content343)
     (allergic_gluten child98)
     (not_allergic_gluten child469)
     (not_allergic_gluten child57)
     (waiting child98 table99)
     (waiting child469 table409)
     (waiting child57 table4)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child98)
     (served child469)
     (served child57)
    )
  )
)
