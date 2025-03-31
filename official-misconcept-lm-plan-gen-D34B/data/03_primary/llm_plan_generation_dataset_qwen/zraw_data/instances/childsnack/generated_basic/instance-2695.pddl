; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 899327

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child482 child312 child488 - child
    bread303 bread233 bread130 - bread-portion
    content127 content90 content157 - content-portion
    tray412 tray306 tray127 - tray
    table447 table352 table38 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray412 kitchen)
     (at tray306 kitchen)
     (at tray127 kitchen)
     (at_kitchen_bread bread303)
     (at_kitchen_bread bread233)
     (at_kitchen_bread bread130)
     (at_kitchen_content content127)
     (at_kitchen_content content90)
     (at_kitchen_content content157)
     (not_allergic_gluten child482)
     (not_allergic_gluten child312)
     (not_allergic_gluten child488)
     (waiting child482 table38)
     (waiting child312 table352)
     (waiting child488 table447)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child482)
     (served child312)
     (served child488)
    )
  )
)
