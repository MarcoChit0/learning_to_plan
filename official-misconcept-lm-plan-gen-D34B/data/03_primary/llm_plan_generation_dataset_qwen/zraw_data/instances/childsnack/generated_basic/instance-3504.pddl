; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 339129

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child453 child186 - child
    bread384 bread165 - bread-portion
    content87 content129 - content-portion
    tray77 tray333 - tray
    table71 table271 table196 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray77 kitchen)
     (at tray333 kitchen)
     (at_kitchen_bread bread384)
     (at_kitchen_bread bread165)
     (at_kitchen_content content87)
     (at_kitchen_content content129)
     (no_gluten_bread bread384)
     (no_gluten_content content129)
     (allergic_gluten child453)
     (not_allergic_gluten child186)
     (waiting child453 table196)
     (waiting child186 table196)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child453)
     (served child186)
    )
  )
)
