; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 401228

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child402 child55 - child
    bread460 bread273 - bread-portion
    content239 content216 - content-portion
    tray410 tray312 tray435 - tray
    table174 table429 table485 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray410 kitchen)
     (at tray312 kitchen)
     (at tray435 kitchen)
     (at_kitchen_bread bread460)
     (at_kitchen_bread bread273)
     (at_kitchen_content content239)
     (at_kitchen_content content216)
     (no_gluten_bread bread460)
     (no_gluten_content content239)
     (allergic_gluten child55)
     (not_allergic_gluten child402)
     (waiting child402 table429)
     (waiting child55 table429)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child402)
     (served child55)
    )
  )
)
