; child-snack task with 3 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 824702

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child446 child242 child376 - child
    bread328 bread212 bread361 - bread-portion
    content353 content2 content366 - content-portion
    tray402 tray457 tray225 tray474 - tray
    table347 table343 table287 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray402 kitchen)
     (at tray457 kitchen)
     (at tray225 kitchen)
     (at tray474 kitchen)
     (at_kitchen_bread bread328)
     (at_kitchen_bread bread212)
     (at_kitchen_bread bread361)
     (at_kitchen_content content353)
     (at_kitchen_content content2)
     (at_kitchen_content content366)
     (no_gluten_bread bread212)
     (no_gluten_content content353)
     (allergic_gluten child242)
     (not_allergic_gluten child376)
     (not_allergic_gluten child446)
     (waiting child446 table287)
     (waiting child242 table287)
     (waiting child376 table343)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child446)
     (served child242)
     (served child376)
    )
  )
)
