; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 760766

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child323 child101 child54 - child
    bread394 bread339 bread80 - bread-portion
    content196 content153 content395 - content-portion
    tray175 tray453 tray479 - tray
    table393 table18 table23 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray175 kitchen)
     (at tray453 kitchen)
     (at tray479 kitchen)
     (at_kitchen_bread bread394)
     (at_kitchen_bread bread339)
     (at_kitchen_bread bread80)
     (at_kitchen_content content196)
     (at_kitchen_content content153)
     (at_kitchen_content content395)
     (not_allergic_gluten child54)
     (not_allergic_gluten child101)
     (not_allergic_gluten child323)
     (waiting child323 table393)
     (waiting child101 table18)
     (waiting child54 table18)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child323)
     (served child101)
     (served child54)
    )
  )
)
