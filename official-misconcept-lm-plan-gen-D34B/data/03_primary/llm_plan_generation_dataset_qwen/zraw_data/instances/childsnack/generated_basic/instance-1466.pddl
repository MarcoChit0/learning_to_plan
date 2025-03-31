; child-snack task with 3 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 571488

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child132 child25 child406 - child
    bread56 bread369 bread383 - bread-portion
    content473 content256 content450 - content-portion
    tray170 - tray
    table189 table149 table266 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray170 kitchen)
     (at_kitchen_bread bread56)
     (at_kitchen_bread bread369)
     (at_kitchen_bread bread383)
     (at_kitchen_content content473)
     (at_kitchen_content content256)
     (at_kitchen_content content450)
     (no_gluten_bread bread369)
     (no_gluten_bread bread56)
     (no_gluten_content content473)
     (no_gluten_content content450)
     (allergic_gluten child25)
     (allergic_gluten child406)
     (not_allergic_gluten child132)
     (waiting child132 table266)
     (waiting child25 table149)
     (waiting child406 table149)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child132)
     (served child25)
     (served child406)
    )
  )
)
