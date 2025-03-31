; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 655937

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child418 child491 child51 - child
    bread451 bread90 bread410 - bread-portion
    content290 content17 content467 - content-portion
    tray470 - tray
    table136 table17 table394 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray470 kitchen)
     (at_kitchen_bread bread451)
     (at_kitchen_bread bread90)
     (at_kitchen_bread bread410)
     (at_kitchen_content content290)
     (at_kitchen_content content17)
     (at_kitchen_content content467)
     (not_allergic_gluten child491)
     (not_allergic_gluten child51)
     (not_allergic_gluten child418)
     (waiting child418 table394)
     (waiting child491 table17)
     (waiting child51 table394)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child418)
     (served child491)
     (served child51)
    )
  )
)
