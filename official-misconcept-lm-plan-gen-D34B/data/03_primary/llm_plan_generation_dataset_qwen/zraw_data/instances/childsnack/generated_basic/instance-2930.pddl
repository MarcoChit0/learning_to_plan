; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 206434

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child188 child185 - child
    bread406 bread435 - bread-portion
    content478 content216 - content-portion
    tray425 tray323 tray134 - tray
    table421 table111 table477 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray425 kitchen)
     (at tray323 kitchen)
     (at tray134 kitchen)
     (at_kitchen_bread bread406)
     (at_kitchen_bread bread435)
     (at_kitchen_content content478)
     (at_kitchen_content content216)
     (no_gluten_bread bread406)
     (no_gluten_content content216)
     (allergic_gluten child188)
     (not_allergic_gluten child185)
     (waiting child188 table421)
     (waiting child185 table477)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child188)
     (served child185)
    )
  )
)
