; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 861464

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child265 child142 - child
    bread488 bread452 - bread-portion
    content85 content158 - content-portion
    tray222 tray403 tray102 - tray
    table447 table23 table423 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray222 kitchen)
     (at tray403 kitchen)
     (at tray102 kitchen)
     (at_kitchen_bread bread488)
     (at_kitchen_bread bread452)
     (at_kitchen_content content85)
     (at_kitchen_content content158)
     (not_allergic_gluten child265)
     (not_allergic_gluten child142)
     (waiting child265 table447)
     (waiting child142 table447)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child265)
     (served child142)
    )
  )
)
