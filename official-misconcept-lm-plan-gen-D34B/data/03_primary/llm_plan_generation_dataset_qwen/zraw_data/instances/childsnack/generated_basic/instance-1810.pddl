; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 54383

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child142 child265 - child
    bread455 bread470 - bread-portion
    content228 content395 - content-portion
    tray289 - tray
    table441 table325 table196 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray289 kitchen)
     (at_kitchen_bread bread455)
     (at_kitchen_bread bread470)
     (at_kitchen_content content228)
     (at_kitchen_content content395)
     (no_gluten_bread bread470)
     (no_gluten_content content395)
     (allergic_gluten child142)
     (not_allergic_gluten child265)
     (waiting child142 table325)
     (waiting child265 table441)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child142)
     (served child265)
    )
  )
)
