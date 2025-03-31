; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 262252

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child248 child253 - child
    bread109 bread170 - bread-portion
    content431 content51 - content-portion
    tray50 - tray
    table4 table136 table309 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray50 kitchen)
     (at_kitchen_bread bread109)
     (at_kitchen_bread bread170)
     (at_kitchen_content content431)
     (at_kitchen_content content51)
     (no_gluten_bread bread109)
     (no_gluten_content content431)
     (allergic_gluten child248)
     (not_allergic_gluten child253)
     (waiting child248 table309)
     (waiting child253 table136)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child248)
     (served child253)
    )
  )
)
