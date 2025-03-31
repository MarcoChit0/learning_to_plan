; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 509505

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child499 child158 child248 - child
    bread129 bread135 bread414 - bread-portion
    content476 content30 content383 - content-portion
    tray422 tray282 tray403 tray39 - tray
    table317 table473 table382 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray422 kitchen)
     (at tray282 kitchen)
     (at tray403 kitchen)
     (at tray39 kitchen)
     (at_kitchen_bread bread129)
     (at_kitchen_bread bread135)
     (at_kitchen_bread bread414)
     (at_kitchen_content content476)
     (at_kitchen_content content30)
     (at_kitchen_content content383)
     (not_allergic_gluten child499)
     (not_allergic_gluten child158)
     (not_allergic_gluten child248)
     (waiting child499 table382)
     (waiting child158 table382)
     (waiting child248 table473)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child499)
     (served child158)
     (served child248)
    )
  )
)
